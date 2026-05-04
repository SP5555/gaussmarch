#pragma once
#include <string>

#include "pipeline.h"
#include "../types/gaussian3d.h"
#include "../layers/gauss_activ_layer.h"
#include "../layers/persp_project_layer.h"
#include "../layers/gsplat_rasterize_layer.h"

/**
 * @brief Forward-only Gaussian splatting pipeline for PLY scene viewing.
 *
 * Loads a PLY file, normalizes splat positions to [-1, 1], and renders
 * forward-only (no backward pass, no optimizer).
 */
class SplatRenderer
{
public:
    ~SplatRenderer() {};

    void init(int w, int h);
    void reloadPLY(const std::string &path, float sceneScale = 1.f);

    bool isLoaded() const { return loaded; }

    void render(const glm::mat4 &view, const glm::mat4 &proj, const glm::vec3 &cam_pos);
    void resize(int newWidth, int newHeight);

    float *getOutput();
    uint32_t getVisibleCount();

    int  getMaxSHDegree()       const { return sh_degree; }
    void setActiveSHDegree(int d)     { atv_layer.setSHDegree(d); }

private:
    void loadPLY(const std::string &path, float sceneScale);
    void initLayers();


    /* ---- state ---- */
    int  width     = 0;
    int  height    = 0;
    int  sh_degree = 0;
    bool loaded    = false;

    /* ---- data ---- */
    Gaussian3DParams gaussian_params;

    /* ---- pipeline ---- */
    Pipeline pipeline;

    /* ---- layers ---- */
    GaussActivLayer   atv_layer;
    PerspProjectLayer psp_layer;
    GsplatRasterizeLayer ras_layer;
};
