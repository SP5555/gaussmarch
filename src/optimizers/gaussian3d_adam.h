#pragma once
#include "adam.h"
#include "../types/gaussian3d.h"

/**
 * @brief Learning rates for each Gaussian3D parameter group.
 *
 * These are intentionally separate from AdamParams (beta1/beta2/epsilon)
 * because they are model-specific, whereas Adam hyperparams are not.
 */
struct Gaussian3DAdamConfig {
    float lr_pos     = 1.6e-2f;
    float lr_scale   = 1.0e-3f;
    float lr_rot     = 6.0e-4f;
    float lr_sh_dc   = 6.0e-3f;
    float lr_sh_rest = 6.0e-3f;
    float lr_opacity = 6.0e-3f;
    AdamParams adam  = {};
};

/**
 * @brief Register all Gaussian3D parameter/gradient buffer pairs into an Adam optimizer.
 *
 * Call once after allocating gaussian_params and grads, before the first step().
 * The Adam optimizer stores raw GPU pointers, so the buffers must outlive the optimizer.
 */
inline void registerGaussian3DGroups(
    Adam&                       adam,
    Gaussian3DParams&           params,
    Gaussian3DGrads&            grads,
    const Gaussian3DAdamConfig& cfg = {})
{
    int n = params.count;
    adam.addGroup(params.pos_x,         grads.grad_pos_x,         n, cfg.lr_pos);
    adam.addGroup(params.pos_y,         grads.grad_pos_y,         n, cfg.lr_pos);
    adam.addGroup(params.pos_z,         grads.grad_pos_z,         n, cfg.lr_pos);
    adam.addGroup(params.scale_x,       grads.grad_scale_x,       n, cfg.lr_scale);
    adam.addGroup(params.scale_y,       grads.grad_scale_y,       n, cfg.lr_scale);
    adam.addGroup(params.scale_z,       grads.grad_scale_z,       n, cfg.lr_scale);
    adam.addGroup(params.rot_w,         grads.grad_rot_w,         n, cfg.lr_rot);
    adam.addGroup(params.rot_x,         grads.grad_rot_x,         n, cfg.lr_rot);
    adam.addGroup(params.rot_y,         grads.grad_rot_y,         n, cfg.lr_rot);
    adam.addGroup(params.rot_z,         grads.grad_rot_z,         n, cfg.lr_rot);
    adam.addGroup(params.sh_dc_r,       grads.grad_sh_dc_r,       n, cfg.lr_sh_dc);
    adam.addGroup(params.sh_dc_g,       grads.grad_sh_dc_g,       n, cfg.lr_sh_dc);
    adam.addGroup(params.sh_dc_b,       grads.grad_sh_dc_b,       n, cfg.lr_sh_dc);
    adam.addGroup(params.logit_opacity, grads.grad_logit_opacity, n, cfg.lr_opacity);

    if (params.sh_num_bands > 0) {
        int n_sh = n * params.sh_num_bands;
        adam.addGroup(params.sh_rest_r, grads.grad_sh_rest_r, n_sh, cfg.lr_sh_rest);
        adam.addGroup(params.sh_rest_g, grads.grad_sh_rest_g, n_sh, cfg.lr_sh_rest);
        adam.addGroup(params.sh_rest_b, grads.grad_sh_rest_b, n_sh, cfg.lr_sh_rest);
    }
}
