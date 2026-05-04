#include <algorithm>
#include <iostream>
#include "cxxopts.hpp"
#include "app/viewer_app.h"
#include "utils/logs.h"

int main(int argc, char *argv[])
{
    cxxopts::Options options("viewer", "PLY Scene Viewer");

    options.add_options()
        ("S,scene",  "Path to PLY scene file", cxxopts::value<std::string>()->default_value(""))
        ("s,scale",  "Scene scale",             cxxopts::value<float>()->default_value("1.0"))
        ("c,camera", "Camera mode (arcball, fly)", cxxopts::value<std::string>()->default_value("arcball"))
        ("help",     "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string ply_path   = result["scene"].as<std::string>();
    float scale            = result["scale"].as<float>();
    std::string camera_str = result["camera"].as<std::string>();

    /* ===== Argument Validation ===== */
    if (scale <= 0.f) {
        log_error("Main", "Error: scale must be a positive number.");
        return 1;
    }

    std::transform(camera_str.begin(), camera_str.end(), camera_str.begin(), ::tolower);
    CameraMode camera_mode;
    if      (camera_str == "arcball") camera_mode = CameraMode::Arcball;
    else if (camera_str == "fly")     camera_mode = CameraMode::Fly;
    else {
        log_error("Main", "Error: invalid camera mode. Valid options: arcball, fly.");
        return 1;
    }

    try {
        ViewerApp app(ply_path, scale, camera_mode);
        app.start();
    }
    catch (const std::exception &e) {
        log_error("Main", std::string("Fatal: ") + e.what());
        return 1;
    }

    return 0;
}