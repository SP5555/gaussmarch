#include "gaussian3d_io.h"

#include <cuda_runtime.h>
#include "../cuda/cuda_check.h"

void uploadGaussians(Gaussian3DParams& params,
                     const std::vector<Gaussian3D>& host,
                     int sh_degree)
{
    int n = (int)host.size();
    params.allocate(n, sh_degree);

    std::vector<float> tmp(n);
    auto up = [&](float* dst, auto getter) {
        for (int i = 0; i < n; i++) tmp[i] = getter(host[i]);
        CUDA_CHECK(cudaMemcpy(dst, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    up(params.pos_x,         [](const Gaussian3D& g) { return g.pos_x; });
    up(params.pos_y,         [](const Gaussian3D& g) { return g.pos_y; });
    up(params.pos_z,         [](const Gaussian3D& g) { return g.pos_z; });
    up(params.scale_x,       [](const Gaussian3D& g) { return g.scale_x; });
    up(params.scale_y,       [](const Gaussian3D& g) { return g.scale_y; });
    up(params.scale_z,       [](const Gaussian3D& g) { return g.scale_z; });
    up(params.rot_w,         [](const Gaussian3D& g) { return g.rot_w; });
    up(params.rot_x,         [](const Gaussian3D& g) { return g.rot_x; });
    up(params.rot_y,         [](const Gaussian3D& g) { return g.rot_y; });
    up(params.rot_z,         [](const Gaussian3D& g) { return g.rot_z; });
    up(params.sh_dc_r,       [](const Gaussian3D& g) { return g.sh_dc_r; });
    up(params.sh_dc_g,       [](const Gaussian3D& g) { return g.sh_dc_g; });
    up(params.sh_dc_b,       [](const Gaussian3D& g) { return g.sh_dc_b; });
    up(params.logit_opacity, [](const Gaussian3D& g) { return g.logit_opacity; });

    // PLY channel-first layout: sh_rest[0..K-1] = R bands, [K..2K-1] = G, [2K..3K-1] = B
    // Our GPU layout: sh_rest_r[band * n + i], so copy band-by-band.
    if (params.sh_num_bands > 0) {
        std::vector<float> band(n);
        for (int b = 0; b < params.sh_num_bands; b++) {
            for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[b];
            CUDA_CHECK(cudaMemcpy(params.sh_rest_r.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));

            for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[params.sh_num_bands + b];
            CUDA_CHECK(cudaMemcpy(params.sh_rest_g.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));

            for (int i = 0; i < n; i++) band[i] = host[i].sh_rest[2 * params.sh_num_bands + b];
            CUDA_CHECK(cudaMemcpy(params.sh_rest_b.ptr + b * n, band.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

std::vector<Gaussian3D> downloadGaussians(const Gaussian3DParams& params)
{
    int n = params.count;
    std::vector<Gaussian3D> host(n);
    std::vector<float> tmp(n);

    auto dn = [&](const float* src, auto setter) {
        CUDA_CHECK(cudaMemcpy(tmp.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) setter(host[i], tmp[i]);
    };

    dn(params.pos_x,         [](Gaussian3D& g, float v) { g.pos_x   = v; });
    dn(params.pos_y,         [](Gaussian3D& g, float v) { g.pos_y   = v; });
    dn(params.pos_z,         [](Gaussian3D& g, float v) { g.pos_z   = v; });
    dn(params.scale_x,       [](Gaussian3D& g, float v) { g.scale_x = v; });
    dn(params.scale_y,       [](Gaussian3D& g, float v) { g.scale_y = v; });
    dn(params.scale_z,       [](Gaussian3D& g, float v) { g.scale_z = v; });
    dn(params.rot_w,         [](Gaussian3D& g, float v) { g.rot_w   = v; });
    dn(params.rot_x,         [](Gaussian3D& g, float v) { g.rot_x   = v; });
    dn(params.rot_y,         [](Gaussian3D& g, float v) { g.rot_y   = v; });
    dn(params.rot_z,         [](Gaussian3D& g, float v) { g.rot_z   = v; });
    dn(params.sh_dc_r,       [](Gaussian3D& g, float v) { g.sh_dc_r = v; });
    dn(params.sh_dc_g,       [](Gaussian3D& g, float v) { g.sh_dc_g = v; });
    dn(params.sh_dc_b,       [](Gaussian3D& g, float v) { g.sh_dc_b = v; });
    dn(params.logit_opacity, [](Gaussian3D& g, float v) { g.logit_opacity = v; });

    if (params.sh_num_bands > 0) {
        for (int b = 0; b < params.sh_num_bands; b++) {
            CUDA_CHECK(cudaMemcpy(tmp.data(), params.sh_rest_r.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) host[i].sh_rest[b] = tmp[i];

            CUDA_CHECK(cudaMemcpy(tmp.data(), params.sh_rest_g.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) host[i].sh_rest[params.sh_num_bands + b] = tmp[i];

            CUDA_CHECK(cudaMemcpy(tmp.data(), params.sh_rest_b.ptr + b * n, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; i++) host[i].sh_rest[2 * params.sh_num_bands + b] = tmp[i];
        }
    }

    return host;
}
