#pragma once
#include <string>
#include <vector>

// Per-Gaussian data loaded from a VEG PLY file.
// scale_x/y/z are in log-space (as stored). opacity and scalar are already sigmoid'd to [0,1].
struct VegGaussian3D {
    float pos_x,   pos_y,   pos_z;
    float scale_x, scale_y, scale_z;    // log-scale
    float rot_w,   rot_x,   rot_y,   rot_z;
    float opacity;   // sigmoid'd
    float scalar;    // sigmoid'd, [0,1]
};

struct VegPLYLoader {
    static std::vector<VegGaussian3D> load(const std::string &path);
};
