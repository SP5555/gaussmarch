#include "raw_volume_io.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <sstream>

// Parses dimensions from filename convention: anything_DxDxD_float32.raw
static bool parse_dims(const std::string &path, int &dx, int &dy, int &dz)
{
    // find last path separator
    size_t sep = path.find_last_of("/\\");
    std::string fname = (sep == std::string::npos) ? path : path.substr(sep + 1);

    // strip extension
    size_t dot = fname.rfind('.');
    if (dot != std::string::npos) fname = fname.substr(0, dot);

    // split by '_'
    std::vector<std::string> parts;
    std::stringstream ss(fname);
    std::string tok;
    while (std::getline(ss, tok, '_')) parts.push_back(tok);

    // look for a token matching DxDxD
    for (const auto &p : parts) {
        int x, y, z;
        if (sscanf(p.c_str(), "%dx%dx%d", &x, &y, &z) == 3) {
            dx = x; dy = y; dz = z;
            return true;
        }
    }
    return false;
}

RawVolume load_raw_volume(const std::string &path)
{
    int dx, dy, dz;
    if (!parse_dims(path, dx, dy, dz))
        throw std::runtime_error("Cannot parse dimensions from filename: " + path +
                                 "\nExpected format: name_DxDxD_float32.raw");

    size_t n = (size_t)dx * dy * dz;

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open volume file: " + path);

    std::vector<float> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("Failed to read volume data from: " + path);

    // Normalize to [0,1]
    float vmin = *std::min_element(raw.begin(), raw.end());
    float vmax = *std::max_element(raw.begin(), raw.end());
    float range = (vmax > vmin) ? (vmax - vmin) : 1.f;
    for (auto &v : raw)
        v = (v - vmin) / range;

    RawVolume vol;
    vol.data  = std::move(raw);
    vol.dim_x = dx;
    vol.dim_y = dy;
    vol.dim_z = dz;
    return vol;
}
