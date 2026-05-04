#pragma once
#include <cstring>
#include <random>
#include <vector>
#include "../types/gaussian3d.h"
#include "sh_consts.h"

namespace SplatUtils
{
    std::vector<Gaussian3D> randomInit(int count, int width, int height, int seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::uniform_real_distribution<float> distu(0.f, 1.f);
        auto rnd  = [&]() { return dist(rng); };
        auto rndu = [&]() { return distu(rng); };

        float half_w    = (float)width  * 0.5f;
        float half_h    = (float)height * 0.5f;
        float log_sigma = logf(2.f);

        std::vector<Gaussian3D> splats(count);
        for (auto &g : splats)
        {
            memset(g.sh_rest, 0, sizeof(g.sh_rest));  // higher-order SH starts at zero

            g.pos_x = rnd() * half_w;
            g.pos_y = rnd() * half_h;
            g.pos_z = rnd() * std::min(half_w, half_h) * 0.25f;

            float s = rnd() * 0.5;
            g.scale_x = log_sigma + s;
            g.scale_y = log_sigma + s;
            g.scale_z = log_sigma + s;

            g.rot_w = 1.f;
            g.rot_x = 0.f;
            g.rot_y = 0.f;
            g.rot_z = 0.f;

            // initialize DC SH coefficients to random colors in [0, 1]
            g.sh_dc_r = (rndu() - 0.5f) / SH_C0;
            g.sh_dc_g = (rndu() - 0.5f) / SH_C0;
            g.sh_dc_b = (rndu() - 0.5f) / SH_C0;

            float o_raw = 0.6f + 0.4f * rndu();
            g.logit_opacity = logf(o_raw / (1.f - o_raw));
        }

        return splats;
    }
    
    void normalizeScene(std::vector<Gaussian3D> &splats, float scene_scale = 1.f)
    {
        // compute centroid
        float cx = 0.f, cy = 0.f, cz = 0.f;
        for (const auto &g : splats)
        {
            cx += g.pos_x;
            cy += g.pos_y;
            cz += g.pos_z;
        }
        float inv = 1.f / (float)splats.size();
        cx *= inv; cy *= inv; cz *= inv;

        // translate to centroid, find max extent
        float max_ext = 0.f;
        for (auto &g : splats)
        {
            g.pos_x -= cx; g.pos_y -= cy; g.pos_z -= cz;

            max_ext = std::max(max_ext, std::abs(g.pos_x));
            max_ext = std::max(max_ext, std::abs(g.pos_y));
            max_ext = std::max(max_ext, std::abs(g.pos_z));
        }

        // scale to [-1, 1]
        if (max_ext > 0.f)
        {
            float linear_scale = scene_scale / max_ext;
            float log_scale = logf(linear_scale);
            for (auto &g : splats)
            {
                g.pos_x *= linear_scale;
                g.pos_y *= linear_scale;
                g.pos_z *= linear_scale;

                g.scale_x += log_scale;
                g.scale_y += log_scale;
                g.scale_z += log_scale;
            }
        }
    }
}
