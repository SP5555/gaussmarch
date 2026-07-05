// gaussmarch_snap: headless offline renderer for VEG Gaussian scenes.
//
// Reads a JSON config (camera, transfer function, render params) and writes
// a single PNG accumulated over N frames. No window, no UI.
//
// Usage: gaussmarch_snap --config render.json [--scene override.ply]

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include "pipelines/gaussian_renderer.h"
#include "io/image_io.h"
#include "cuda/cuda_check.h"
#include "optix/optix_check.h"
#include "utils/logs.h"
#include "utils/ansi_colors.h"

using json = nlohmann::json;

// Linear interpolation of TF control points into a flat RGBA array of n entries.
// Control points: [{"t":0.0,"r":...,"g":...,"b":...,"a":...}, ...]
static std::vector<float> build_colormap(const json &tf_points, int n = 256)
{
    struct CP { float t, r, g, b, a; };
    std::vector<CP> cps;
    for (auto &p : tf_points)
        cps.push_back({p["t"], p["r"], p["g"], p["b"], p["a"]});
    std::sort(cps.begin(), cps.end(), [](const CP &a, const CP &b){ return a.t < b.t; });

    if (cps.empty()) throw std::runtime_error("transfer_function must have at least one control point");

    std::vector<float> cm(n * 4);
    for (int i = 0; i < n; ++i) {
        float t = (float)i / (float)(n - 1);

        // Find surrounding control points
        int lo = 0, hi = (int)cps.size() - 1;
        for (int j = 0; j < (int)cps.size() - 1; ++j) {
            if (cps[j].t <= t) { lo = j; hi = j + 1; }
        }

        float alpha = 0.f;
        if (cps[hi].t > cps[lo].t)
            alpha = (t - cps[lo].t) / (cps[hi].t - cps[lo].t);
        alpha = glm::clamp(alpha, 0.f, 1.f);

        cm[i*4+0] = cps[lo].r + alpha * (cps[hi].r - cps[lo].r);
        cm[i*4+1] = cps[lo].g + alpha * (cps[hi].g - cps[lo].g);
        cm[i*4+2] = cps[lo].b + alpha * (cps[hi].b - cps[lo].b);
        cm[i*4+3] = cps[lo].a + alpha * (cps[hi].a - cps[lo].a);
    }
    return cm;
}

int main(int argc, char **argv)
{
    try {
        cxxopts::Options opts("gaussmarch_snap", "Headless offline renderer for VEG Gaussian scenes");
        opts.add_options()
            ("c,config", "Path to render config JSON", cxxopts::value<std::string>())
            ("scene",    "Override scene path from config", cxxopts::value<std::string>()->default_value(""))
            ("h,help",   "Print help");

        auto args = opts.parse(argc, argv);
        if (args.count("help") || !args.count("config")) {
            std::cout << opts.help() << "\n";
            return 0;
        }

        // ===== Load config =====
        std::string config_path = args["config"].as<std::string>();
        std::ifstream f(config_path);
        if (!f) throw std::runtime_error("Cannot open config: " + config_path);
        json cfg = json::parse(f);

        // Scene
        std::string scene_path = args["scene"].as<std::string>();
        if (scene_path.empty()) scene_path = cfg.value("scene", "");
        if (scene_path.empty()) throw std::runtime_error("No scene specified -- add \"scene\" to config or use --scene");

        // Output
        std::string output_path  = cfg.value("output",              "render.png");
        int         width        = cfg.value("width",               1920);
        int         height       = cfg.value("height",              1080);
        int         accum_frames = cfg.value("accumulation_frames", 256);

        // Camera
        auto &cam  = cfg.at("camera");
        glm::vec3 pos  = {cam["pos"][0],  cam["pos"][1],  cam["pos"][2]};
        glm::vec3 look = {cam["look"][0], cam["look"][1], cam["look"][2]};
        glm::vec3 up   = {cam["up"][0],   cam["up"][1],   cam["up"][2]};
        float fov = cam.value("fov", 45.f);

        // glm::lookAt orthogonalizes 'up' against the view direction automatically.
        glm::mat4 view = glm::lookAt(pos, look, up);
        glm::mat4 proj = glm::perspective(glm::radians(fov), (float)width / height, 0.01f, 100.f);

        // Render params
        json r = cfg.value("render", json::object());
        float step_size    = r.value("step_size",       0.01f);
        float shadow_step  = r.value("shadow_step",     0.02f);
        int   max_depth    = r.value("max_depth",       400);
        bool  shadows      = r.value("shadows",         true);
        float ambient      = r.value("ambient",         0.4f);
        float az_deg       = r.value("light_azimuth",   40.f);   // degrees 0-360
        float el_deg       = r.value("light_elevation", 30.f);   // degrees -90 to 90
        bool  blue_noise   = r.value("blue_noise",      true);
        float scene_scale  = r.value("scale",           1.f);

        // ===== CUDA + OptiX init (no window required) =====
        PRINT_BUILD_INFO();
        log_info("gaussmarch_snap", "Config: " + config_path, ANSI_MAGENTA);
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        log_info("gaussmarch_snap", std::string("CUDA Device: ") + deviceProp.name, ANSI_MAGENTA);
        OPTIX_CHECK(optixInit());
        CUcontext cu_ctx = 0;  // current CUDA context
        OptixDeviceContextOptions optix_opts = {};
        OptixDeviceContext optix_ctx;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &optix_opts, &optix_ctx));

        // ===== Renderer =====
        GaussianRenderer renderer;
        renderer.init(optix_ctx, width, height);
        renderer.loadGaussians(scene_path, scene_scale);

        renderer.step_size          = step_size;
        renderer.shadow_step_size   = shadow_step;
        renderer.max_depth          = max_depth;
        renderer.shadows_enabled    = shadows;
        renderer.light_ambient      = ambient;
        renderer.light_azimuth      = az_deg / 360.f;
        renderer.light_elevation    = el_deg / 180.f + 0.5f;
        renderer.blue_noise_enabled = blue_noise;
        renderer.accum_enabled      = true;

        auto colormap = build_colormap(cfg.at("transfer_function"));
        renderer.setColormap(colormap.data(), (int)colormap.size() / 4);

        // ===== Accumulation render loop =====
        std::cout << "Rendering " << accum_frames << " accumulation frames at "
                  << width << "x" << height << "\n";
        const int bar_width = 40;
        for (int i = 0; i < accum_frames; ++i) {
            renderer.render(view, proj, pos);

            int done  = (i + 1) * bar_width / accum_frames;
            int pct   = (i + 1) * 100 / accum_frames;
            std::cout << "\r  [";
            for (int b = 0; b < bar_width; ++b)
                std::cout << (b < done ? '#' : '.');
            std::cout << "] " << (i + 1) << "/" << accum_frames
                      << " (" << pct << "%)   " << std::flush;
        }
        std::cout << "\n";

        // ===== Save =====
        ImageSaver::saveAsPNG(renderer.getOutput(), width, height, output_path);
        std::cout << "Saved: " << output_path << "\n";

        optixDeviceContextDestroy(optix_ctx);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
