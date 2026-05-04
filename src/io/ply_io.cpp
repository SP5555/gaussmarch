#include "ply_io.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "../utils/logs.h"

/* ===== ===== PLY Loader ===== ===== */

PLYLoadResult PLYLoader::load(const std::string &path)
{
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != "ply")
        log_fatal("PLYLoader", "Unsupported file extension: " + ext);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        log_fatal("PLYLoader", "Failed to open: " + path);

    // ===== parse header =====
    std::string line;
    int vertex_count = 0;
    std::vector<std::string> property_order;
    bool header_done = false;

    while (std::getline(file, line))
    {
        // strip trailing \r for Windows-style line endings
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.find("element vertex") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp;
            ss >> tmp >> tmp >> vertex_count;
        }
        else if (line.find("property float") != std::string::npos)
        {
            std::istringstream ss(line);
            std::string tmp, name;
            ss >> tmp >> tmp >> name;
            property_order.push_back(name);
        }
        else if (line == "end_header")
        {
            header_done = true;
            break;
        }
    }

    if (!header_done)
        log_fatal("PLYLoader", "Malformed PLY: end_header not found");
    if (vertex_count <= 0)
        log_fatal("PLYLoader", "Malformed PLY: no vertices found");

    // ===== build property index map =====
    std::unordered_map<std::string, int> idx;
    for (int i = 0; i < (int)property_order.size(); i++)
        idx[property_order[i]] = i;

    auto require = [&](const std::string &name) {
        if (idx.find(name) == idx.end())
            log_fatal("PLYLoader", "Missing property: " + name);
        return idx[name];
    };

    const int i_posx    = require("x");
    const int i_posy    = require("y");
    const int i_posz    = require("z");
    const int i_scale0  = require("scale_0");
    const int i_scale1  = require("scale_1");
    const int i_scale2  = require("scale_2");
    const int i_rot0    = require("rot_0");
    const int i_rot1    = require("rot_1");
    const int i_rot2    = require("rot_2");
    const int i_rot3    = require("rot_3");
    const int i_dc0     = require("f_dc_0");
    const int i_dc1     = require("f_dc_1");
    const int i_dc2     = require("f_dc_2");
    const int i_opacity = require("opacity");

    // ===== detect SH degree from f_rest_* count =====
    int f_rest_count = 0;
    for (const auto &name : property_order)
        if (name.rfind("f_rest_", 0) == 0) f_rest_count++;

    // channel-first layout: 3 channels per band set
    int sh_num_bands = f_rest_count / 3;
    int sh_degree    = 0;
    if      (sh_num_bands >= 15) { sh_degree = 3; sh_num_bands = 15; }
    else if (sh_num_bands >= 8 ) { sh_degree = 2; sh_num_bands = 8;  }
    else if (sh_num_bands >= 3 ) { sh_degree = 1; sh_num_bands = 3;  }

    log_info("PLYLoader", "Detected SH degree: " + std::to_string(sh_degree) +
             " (" + std::to_string(sh_num_bands * 3) + " f_rest properties used)");

    // map f_rest_* indices (only up to what we'll use)
    std::vector<int> i_rest(sh_num_bands * 3, -1);
    for (int b = 0; b < sh_num_bands * 3; b++)
    {
        std::string key = "f_rest_" + std::to_string(b);
        if (idx.count(key)) i_rest[b] = idx[key];
    }

    // ===== read binary data =====
    const int stride = (int)property_order.size() * sizeof(float);
    std::vector<float> row(property_order.size());

    std::vector<Gaussian3D> splats;
    splats.reserve(vertex_count);

    for (int i = 0; i < vertex_count; i++)
    {
        file.read(reinterpret_cast<char *>(row.data()), stride);
        if (!file)
            log_fatal("PLYLoader", "Unexpected end of file at vertex " + std::to_string(i));

        Gaussian3D g;

        g.pos_x = row[i_posx];
        g.pos_y = row[i_posy];
        g.pos_z = row[i_posz];

        g.scale_x = row[i_scale0];
        g.scale_y = row[i_scale1];
        g.scale_z = row[i_scale2];

        // PLY stores (w, x, y, z) as (rot_0, rot_1, rot_2, rot_3)
        g.rot_w = row[i_rot0];
        g.rot_x = row[i_rot1];
        g.rot_y = row[i_rot2];
        g.rot_z = row[i_rot3];

        // PLY files from 3DGS training are in OpenCV convention (X right, Y down, Z forward).
        // Convert to OpenGL convention (X right, Y up, Z backward) by flipping both Y and Z.
        // Quaternion for combined Y+Z flip: negate y and z components (w cancels from double negation).
        g.pos_y = -g.pos_y;
        g.pos_z = -g.pos_z;
        g.rot_y = -g.rot_y;
        g.rot_z = -g.rot_z;

        // DC SH coefficients
        g.sh_dc_r = row[i_dc0];
        g.sh_dc_g = row[i_dc1];
        g.sh_dc_b = row[i_dc2];

        // higher-order SH: sh_rest layout is channel-first
        // [0..K-1] = R bands, [K..2K-1] = G bands, [2K..3K-1] = B bands
        memset(g.sh_rest, 0, sizeof(g.sh_rest));
        for (int b = 0; b < sh_num_bands * 3; b++)
            if (i_rest[b] >= 0) g.sh_rest[b] = row[i_rest[b]];

        // logit-opacity stored as-is, sigmoid applied in GaussActivLayer
        g.logit_opacity = row[i_opacity];

        splats.push_back(g);
    }

    log_info("PLYLoader", "Loaded " + std::to_string(vertex_count) + " splats from " + path);
    return { std::move(splats), sh_degree };
}

/* ===== ===== PLY Saver ===== ===== */

void PLYSaver::save(const std::string &path, const std::vector<Gaussian3D> &splats,
                    int sh_num_bands)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        log_fatal("PLYSaver", "Failed to open for writing: " + path);

    int n = (int)splats.size();

    // ===== write header =====
    // property count: x y z  f_dc(3)  f_rest(3*K)  opacity  scale(3)  rot(4)
    int n_rest = sh_num_bands * 3;

    auto prop = [&](const std::string &name) {
        file << "property float " << name << "\n";
    };

    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << n << "\n";
    prop("x"); prop("y"); prop("z");
    prop("f_dc_0"); prop("f_dc_1"); prop("f_dc_2");
    for (int i = 0; i < n_rest; i++)
        prop("f_rest_" + std::to_string(i));
    prop("opacity");
    prop("scale_0"); prop("scale_1"); prop("scale_2");
    prop("rot_0"); prop("rot_1"); prop("rot_2"); prop("rot_3");
    file << "end_header\n";

    // ===== write binary data =====
    for (const auto &g : splats)
    {
        // Convert OpenGL (Y up, Z backward) back to OpenCV (Y down, Z forward)
        // by flipping Y and Z. Quaternion: negate rot_y and rot_z
        // (rot_w double-negation from Y and Z flips cancels out).
        float pos_x =  g.pos_x;
        float pos_y = -g.pos_y;
        float pos_z = -g.pos_z;
        float rot_w =  g.rot_w;
        float rot_x =  g.rot_x;
        float rot_y = -g.rot_y;
        float rot_z = -g.rot_z;

        auto wf = [&](float v) { file.write(reinterpret_cast<const char *>(&v), sizeof(float)); };

        wf(pos_x); wf(pos_y); wf(pos_z);
        wf(g.sh_dc_r); wf(g.sh_dc_g); wf(g.sh_dc_b);
        for (int i = 0; i < n_rest; i++) wf(g.sh_rest[i]);
        wf(g.logit_opacity);
        wf(g.scale_x); wf(g.scale_y); wf(g.scale_z);
        // PLY convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z
        wf(rot_w); wf(rot_x); wf(rot_y); wf(rot_z);
    }

    log_info("PLYSaver", "Saved " + std::to_string(n) + " splats to " + path);
}
