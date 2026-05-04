#pragma once

#include <cuda_runtime.h>

#include <string>

#include "pipeline.h"
#include "../layers/gauss_activ_layer.h"
#include "../layers/mse_loss_layer.h"
#include "../layers/persp_project_layer.h"
#include "../layers/gsplat_rasterize_layer.h"
#include "../optimizers/gaussian3d_adam.h"
#include "../types/gaussian3d.h"

/**
 * @brief Differentiable Gaussian image fitter.
 *
 * Pure CUDA pipeline, no GL, no display, no window.
 * Owns Gaussian parameters and all layers.
 * Call render() each frame, then getOutput() to get the pixel buffer
 * for display via AppBase::displayFrame().
 */
class ImageFitter
{
public:
    ~ImageFitter() {};

    void init(int width, int height);
    void loadTargetImage(const std::string &imagePath, int w, int h, int padding = 0);
    void randomInitGaussians(int count, int seed = -1);
    void initLayers();

    void step();
    void savePLY(const std::string &path);

    float *getOutput();
    int    getIterCount() const { return optimizer.getStepCount(); }
    float  getLoss() const      { return mse_layer.getLoss(); }

    bool is_optimization_running = true;
private:
    /* ---- config ---- */
    int width  = 0;
    int height = 0;


    /* ---- Gaussian state ---- */
    Gaussian3DParams gaussian_params;

    /* ---- pipeline ---- */
    Pipeline pipeline;

    /* ---- layers ---- */
    GaussActivLayer   atv_layer;
    PerspProjectLayer psp_layer;
    GsplatRasterizeLayer ras_layer;
    MSELossLayer      mse_layer;

    /* ---- optimizer ---- */
    Adam optimizer;

    /* ---- target image ---- */
    CudaBuffer<float> d_target_pixels;
};
