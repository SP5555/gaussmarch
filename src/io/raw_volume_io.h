#pragma once
#include <string>
#include <vector>
#include <stdint.h>

struct RawVolume {
    std::vector<float> data;   // normalized [0,1] float values
    int dim_x, dim_y, dim_z;
};

// Loads a raw binary float32 volume file.
// Filename must follow the convention: name_DxDxD_float32.raw
// Values are normalized to [0,1].
RawVolume load_raw_volume(const std::string &path);
