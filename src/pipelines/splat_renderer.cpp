#include "splat_renderer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../cuda/cuda_check.h"
#include "../io/ply_io.h"
#include "../utils/gaussian3d_io.h"
#include "../utils/logs.h"
#include "../utils/splat_utils.h"

/* ===== ===== Lifecycle ===== ===== */

void SplatRenderer::init(int w, int h)
{
    width       = w;
    height      = h;

    log_info("SplatRenderer",
        "WindowSize=" + std::to_string(w) + "x" + std::to_string(h)
    );
}

void SplatRenderer::loadPLY(const std::string &path, const float sceneScale)
{
    auto result = PLYLoader::load(path);
    if (result.splats.empty())
        throw std::runtime_error("[SplatRenderer] PLY loaded 0 splats: " + path);

    SplatUtils::normalizeScene(result.splats, sceneScale);
    sh_degree = result.sh_degree;
    uploadGaussians(gaussian_params, result.splats, sh_degree);
}

float *SplatRenderer::getOutput()
{
    return ras_layer.getOutput();
}

uint32_t SplatRenderer::getVisibleCount()
{
    return ras_layer.getVisibleCount();
}

/* ===== ===== Init ===== ===== */

void SplatRenderer::initLayers()
{
    int count = gaussian_params.count;

    // allocate forward buffers only (no grad buffers needed for viewer)
    atv_layer.setSHDegree(sh_degree);
    atv_layer.allocate(count);
    psp_layer.allocate(count);
    ras_layer.allocate(width, height, count);

    // wire forward
    atv_layer.setInput(&gaussian_params);
    psp_layer.setInput(&atv_layer.getOutput());
    ras_layer.setInput(&psp_layer.getOutput());

    // no backward wiring, forward-only pipeline

    // register in pipeline
    pipeline.add(&atv_layer);
    pipeline.add(&psp_layer);
    pipeline.add(&ras_layer);

    loaded = true;
}

void SplatRenderer::reloadPLY(const std::string &path, float sceneScale)
{
    loaded = false;
    pipeline.clear();
    loadPLY(path, sceneScale);
    initLayers();
}

/* ===== ===== Render ===== ===== */

void SplatRenderer::render(const glm::mat4 &view, const glm::mat4 &proj, const glm::vec3 &cam_pos)
{
    atv_layer.setCameraPosition(cam_pos);
    psp_layer.setCamera(view, proj);

    pipeline.forward();
}

void SplatRenderer::resize(int newWidth, int newHeight)
{
    width  = newWidth;
    height = newHeight;

    ras_layer.resize(width, height);
}
