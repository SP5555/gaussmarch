#pragma once
#include <vector>
#include "../types/gaussian3d.h"

/**
 * @brief CPU <-> GPU transfer for Gaussian3DParams.
 *
 * Separated from Gaussian3DParams so the GPU type stays a pure memory layout.
 * Future types follow the same pattern: add a companion *_io.h in src/utils/.
 */
void uploadGaussians(Gaussian3DParams& params,
                     const std::vector<Gaussian3D>& host,
                     int sh_degree = 0);

std::vector<Gaussian3D> downloadGaussians(const Gaussian3DParams& params);
