#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

static constexpr int TF_TEXTURE_SIZE = 256;

// ===== ColormapTF =====
// 1D RGBA transfer function edited via a gradient bar + draggable stop markers.
// Each stop carries RGBA; the bar interpolates linearly between stops.

struct ColormapPoint {
    float t;
    float r, g, b, a;
};

class ColormapTF {
public:
    ColormapTF();
    ~ColormapTF();

    // Draw gradient editor inline (no Begin/End needed).
    // Returns true when the TF changed this frame.
    bool draw(const char *label);

    void upload();
    cudaTextureObject_t texture() const { return tex_obj; }
    const std::vector<ColormapPoint> &points() const { return pts; }

private:
    void rebuildTexture();
    float4 sample(float t) const;

    std::vector<ColormapPoint> pts;
    bool dirty      = true;
    int  selected   = -1;
    int  drag_idx   = -1;

    cudaArray_t         tex_array = nullptr;
    cudaTextureObject_t tex_obj   = 0;
};

// ===== ScalarTF =====
// 1D scalar [0,1] transfer function (density or radius multiplier).
// Same gradient-bar UI, but stops only carry a scalar value; bar shown as grayscale.

struct ScalarPoint {
    float t;
    float v;
};

class ScalarTF {
public:
    explicit ScalarTF(float default_value = 1.f);
    ~ScalarTF();

    bool draw(const char *label);
    void upload();

    cudaTextureObject_t texture() const { return tex_obj; }
    const std::vector<ScalarPoint> &points() const { return pts; }

private:
    void rebuildTexture();
    float sample(float t) const;

    std::vector<ScalarPoint> pts;
    bool dirty    = true;
    int  selected = -1;
    int  drag_idx = -1;

    cudaArray_t         tex_array = nullptr;
    cudaTextureObject_t tex_obj   = 0;
};
