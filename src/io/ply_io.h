#pragma once
#include <string>
#include <vector>

#include "../types/gaussian3d.h"

/**
 * @brief Loads a 3DGS PLY file into a vector of Gaussian3D structs.
 *
 * Expects binary little-endian PLY format as output by standard
 * 3D Gaussian Splatting training (e.g. gaussian-splatting, gsplat).
 *
 * Conversions applied:
 *   - Color: f_dc stored as-is (DC SH coefficient, activation in GaussActivLayer)
 *   - Scale: stored as-is     (already log-scale, exp applied in GaussActivLayer)
 *   - Opacity: stored as-is   (already logit, sigmoid applied in GaussActivLayer)
 *   - Rotation: rot_0 = w, rot_1 = x, rot_2 = y, rot_3 = z
 *
 * SH degree is auto-detected from the number of f_rest_* properties:
 *   - 0 properties  -> degree 0 (DC only)
 *   - 9 properties  -> degree 1
 *   - 24 properties -> degree 2
 *   - 45 properties -> degree 3
 */
struct PLYLoadResult
{
    std::vector<Gaussian3D> splats;
    int sh_degree = 0;
};

class PLYLoader
{
public:
    static PLYLoadResult load(const std::string &path);
};

/**
 * @brief Saves Gaussian splats to a binary little-endian PLY file.
 *
 * Splats must be in OpenGL convention (Y up, Z backward) -- the same
 * convention used internally by both apps. Y and Z axes (and the
 * corresponding quaternion components) are flipped on write to produce
 * the OpenCV convention (Y down, Z forward) that standard 3DGS tooling
 * and PLYLoader expect.
 *
 * NOTE: positions are saved in whatever scale they were trained in
 * (e.g. pixel-scaled world space for fitter). PLYLoader normalizes
 * the scene to [-1, 1] on load, so scale differences are handled automatically.
 *
 * Property order written: x y z f_dc_0..2 [f_rest_0..N] opacity scale_0..2 rot_0..3
 */
class PLYSaver
{
public:
    static void save(const std::string &path, const std::vector<Gaussian3D> &splats,
                     int sh_num_bands = 0);
};
