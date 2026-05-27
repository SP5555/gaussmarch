#include "veg_ply_io.h"

#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "../utils/logs.h"

// ===== Property types =====

enum class PropType { Float32, Int16, UInt8, Unknown };

static PropType parsePropType(const std::string &s)
{
    if (s == "float"  || s == "float32") return PropType::Float32;
    if (s == "short"  || s == "int16")   return PropType::Int16;
    if (s == "uchar"  || s == "uint8")   return PropType::UInt8;
    return PropType::Unknown;
}

static int propBytes(PropType t)
{
    switch (t) {
        case PropType::Float32: return 4;
        case PropType::Int16:   return 2;
        case PropType::UInt8:   return 1;
        default:                return 0;
    }
}

// ===== Half-float decoder =====

static float half_to_float(int16_t h)
{
    uint16_t u; memcpy(&u, &h, 2);
    uint32_t sign = (u >> 15) & 0x1u;
    uint32_t exp  = (u >> 10) & 0x1fu;
    uint32_t mant =  u        & 0x3ffu;
    uint32_t f;
    if      (exp == 0)  f = sign << 31;
    else if (exp == 31) f = (sign << 31) | (0xffu << 23) | (mant << 13);
    else                f = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// ===== PLY header =====

struct PlyProp    { std::string name; PropType type; };
struct PlyElement {
    std::string name; int count = 0;
    std::vector<PlyProp> props;
    int stride() const {
        int s = 0; for (auto &p : props) s += propBytes(p.type); return s;
    }
};

static std::vector<PlyElement> parseHeader(std::ifstream &file, const std::string &path)
{
    std::vector<PlyElement> elems;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("element ", 0) == 0) {
            std::istringstream ss(line); std::string tmp, name; int count = 0;
            ss >> tmp >> name >> count;
            elems.push_back({name, count, {}});
        } else if (line.rfind("property ", 0) == 0 && !elems.empty()) {
            std::istringstream ss(line); std::string tmp, type_str, prop_name;
            ss >> tmp >> type_str >> prop_name;
            elems.back().props.push_back({prop_name, parsePropType(type_str)});
        } else if (line == "end_header") { break; }
    }
    return elems;
}

// ===== Codebook =====

using Codebook = std::unordered_map<std::string, std::vector<float>>;

static Codebook readCodebook(std::ifstream &file, const PlyElement &elem)
{
    Codebook cb;
    int stride = elem.stride();
    for (auto &p : elem.props) cb[p.name].resize(elem.count);

    std::vector<uint8_t> row((size_t)stride);
    for (int i = 0; i < elem.count; i++) {
        file.read(reinterpret_cast<char *>(row.data()), stride);
        if (!file) log_fatal("VegPLYLoader", "Unexpected EOF in codebook_centers");
        int off = 0;
        for (auto &p : elem.props) {
            float val = 0.f;
            if      (p.type == PropType::Int16)   { int16_t v; memcpy(&v, row.data()+off, 2); val = half_to_float(v); }
            else if (p.type == PropType::Float32)  { memcpy(&val, row.data()+off, 4); }
            else if (p.type == PropType::UInt8)    { val = (float)row[off]; }
            cb[p.name][i] = val;
            off += propBytes(p.type);
        }
    }
    return cb;
}

// ===== Gaussian reader =====

static std::vector<VegGaussian3D> readGaussians(std::ifstream    &file,
                                                const PlyElement  &elem,
                                                const Codebook    &cb)
{
    std::unordered_map<std::string, int>      offsets;
    std::unordered_map<std::string, PropType> types;
    int off = 0;
    for (auto &p : elem.props) {
        offsets[p.name] = off;
        types[p.name]   = p.type;
        off += propBytes(p.type);
    }
    int stride = elem.stride();

    auto req_off = [&](const std::string &n) {
        auto it = offsets.find(n);
        if (it == offsets.end()) log_fatal("VegPLYLoader", "Missing property: " + n);
        return it->second;
    };

    int ox  = req_off("x"),       oy  = req_off("y"),       oz  = req_off("z");
    int oo  = req_off("opacity");
    int os0 = req_off("scale_0"), os1 = req_off("scale_1"), os2 = req_off("scale_2");
    int or0 = req_off("rot_0"),   or1 = req_off("rot_1"),   or2 = req_off("rot_2"), or3 = req_off("rot_3");
    int osc = req_off("scalar");

    const auto &cb_opacity = cb.at("opacity");
    const auto &cb_scaling = cb.at("scaling");
    const auto &cb_rot_re  = cb.at("rotation_re");
    const auto &cb_rot_im  = cb.at("rotation_im");
    const auto &cb_scalar  = cb.at("scalar");

    std::vector<VegGaussian3D> splats;
    splats.reserve(elem.count);
    std::vector<uint8_t> row((size_t)stride);

    for (int i = 0; i < elem.count; i++) {
        file.read(reinterpret_cast<char *>(row.data()), stride);
        if (!file) log_fatal("VegPLYLoader", "Unexpected EOF at gaussian " + std::to_string(i));

        VegGaussian3D g;

        { int16_t v; memcpy(&v, row.data()+ox, 2); g.pos_x =  half_to_float(v); }
        { int16_t v; memcpy(&v, row.data()+oy, 2); g.pos_y = -half_to_float(v); }  // flip Y
        { int16_t v; memcpy(&v, row.data()+oz, 2); g.pos_z = -half_to_float(v); }  // flip Z

        g.opacity  = sigmoid(cb_opacity[row[oo]]);
        g.scale_x  = cb_scaling[row[os0]];
        g.scale_y  = cb_scaling[row[os1]];
        g.scale_z  = cb_scaling[row[os2]];
        g.rot_w    =  cb_rot_re[row[or0]];
        g.rot_x    =  cb_rot_im[row[or1]];
        g.rot_y    = -cb_rot_im[row[or2]];  // Y flip
        g.rot_z    = -cb_rot_im[row[or3]];  // Z flip
        g.scalar   = sigmoid(cb_scalar[row[osc]]);

        splats.push_back(g);
    }
    return splats;
}

// ===== Public API =====

std::vector<VegGaussian3D> VegPLYLoader::load(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) log_fatal("VegPLYLoader", "Failed to open: " + path);

    auto elems = parseHeader(file, path);

    const PlyElement *gaussians_elem = nullptr;
    const PlyElement *codebook_elem  = nullptr;
    for (auto &e : elems) {
        if (e.name == "gaussians")        gaussians_elem = &e;
        if (e.name == "codebook_centers") codebook_elem  = &e;
    }
    if (!gaussians_elem)
        log_fatal("VegPLYLoader", "No 'gaussians' element in: " + path);
    if (!codebook_elem)
        log_fatal("VegPLYLoader", "No 'codebook_centers' element in: " + path);

    // Skip gaussians section, read codebook, then reopen and read gaussians.
    long long gaussians_bytes = (long long)gaussians_elem->count * gaussians_elem->stride();
    file.seekg(gaussians_bytes, std::ios::cur);
    Codebook codebook = readCodebook(file, *codebook_elem);
    file.close();

    file.open(path, std::ios::binary);
    parseHeader(file, path);

    auto splats = readGaussians(file, *gaussians_elem, codebook);
    log_info("VegPLYLoader", "Loaded " + std::to_string(splats.size()) + " gaussians from " + path);
    return splats;
}
