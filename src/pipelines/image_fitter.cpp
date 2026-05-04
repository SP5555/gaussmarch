#include "image_fitter.h"

#include <cuda_runtime.h>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>

#include "../cuda/cuda_check.h"
#include "../io/image_io.h"
#include "../io/ply_io.h"
#include "../utils/gaussian3d_io.h"
#include "../utils/logs.h"
#include "../utils/splat_utils.h"

/* ===== ===== Lifecycle ===== ===== */

void ImageFitter::init(int w, int h)
{
    width  = w;
    height = h;

    float half_w = width  * 0.5f;
    float half_h = height * 0.5f;
    float z_range = std::min(half_w, half_h) * 0.25f; // matches init range
    psp_layer.setCamera(
        glm::mat4(1.0f),                                          // identity view
        glm::ortho(-half_w, half_w, -half_h, half_h, -z_range, z_range)
    );

    log_info("ImageFitter",
        "WindowSize=" + std::to_string(w) + "x" + std::to_string(h)
    );
}

void ImageFitter::loadTargetImage(const std::string &imagePath, int w, int h, int padding)
{
    auto image = ImageLoader::load(imagePath, w, h, padding);
    if (image.pixels.empty())
        log_fatal("ImageFitter", "Failed to load target image: " + imagePath);

    d_target_pixels.allocate(w * h * 3);
    CUDA_CHECK(cudaMemcpy(d_target_pixels, image.pixels.data(),
                          w * h * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void ImageFitter::randomInitGaussians(int count, int seed)
{
    if (seed < 0)
        seed = (int)std::chrono::system_clock::now().time_since_epoch().count();

    auto splats = SplatUtils::randomInit(count, width, height, seed);
    uploadGaussians(gaussian_params, splats);
}

float *ImageFitter::getOutput()
{
    return ras_layer.getOutput();
}

/* ===== ===== Init ===== ===== */

void ImageFitter::initLayers()
{
    int count = gaussian_params.count;

    // allocate forward buffers
    atv_layer.allocate(count);
    psp_layer.allocate(count);
    ras_layer.allocate(width, height, count);
    mse_layer.allocate(width, height);

    // allocate grad buffers
    atv_layer.allocateGrad(count);
    psp_layer.allocateGrad(count);
    ras_layer.allocateGrad(count);

    // wire forward
    atv_layer.setInput(&gaussian_params);
    psp_layer.setInput(&atv_layer.getOutput());
    ras_layer.setInput(&psp_layer.getOutput());
    mse_layer.setInput(ras_layer.getOutput());
    mse_layer.setTarget(d_target_pixels);

    // wire backward
    ras_layer.setGradOutput(&mse_layer.getGradInput());
    psp_layer.setGradOutput(&ras_layer.getGradInput());
    atv_layer.setGradOutput(&psp_layer.getGradInput());

    // optimizer state
    optimizer.init();
    registerGaussian3DGroups(optimizer, gaussian_params, atv_layer.getGradInput());

    // register in pipeline
    pipeline.add(&atv_layer);
    pipeline.add(&psp_layer);
    pipeline.add(&ras_layer);
    pipeline.add(&mse_layer);
}

/* ===== ===== Render ===== ===== */

void ImageFitter::step()
{
    pipeline.zeroGrad();
    pipeline.forward();

    if (!is_optimization_running) return;

    pipeline.backward();
    optimizer.step();
}

void ImageFitter::savePLY(const std::string &path)
{
    auto splats = downloadGaussians(gaussian_params);
    PLYSaver::save(path, splats, gaussian_params.sh_num_bands);
}
