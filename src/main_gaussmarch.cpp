#include <iostream>
#include <algorithm>
#include <string>

#include "cxxopts.hpp"
#include "app/gaussmarch_app.h"
#include "utils/logs.h"

int main(int argc, char *argv[])
{
    cxxopts::Options options("gaussmarch", "Gaussian volume renderer");
    options.add_options()
        ("s,scene",   "Path to VEG PLY file",
                     cxxopts::value<std::string>()->default_value(""))
        ("scale",    "Scene scale (default 1.0)",
                     cxxopts::value<float>()->default_value("1.0"))
        ("c,camera", "Camera mode: fly | arcball",
                     cxxopts::value<std::string>()->default_value("arcball"))
        ("help",     "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    std::string path    = result["scene"].as<std::string>();
    float       scale   = result["scale"].as<float>();
    std::string cam_str = result["camera"].as<std::string>();
    std::transform(cam_str.begin(), cam_str.end(), cam_str.begin(), ::tolower);

    CameraMode cam_mode = CameraMode::Arcball;
    if (cam_str == "fly") cam_mode = CameraMode::Fly;
    else if (cam_str != "arcball") {
        log_error("Main", "Invalid camera mode. Valid options: fly, arcball.");
        return 1;
    }

    if (scale <= 0.f) {
        log_error("Main", "scale must be a positive number.");
        return 1;
    }

    try {
        GaussmarchApp app(path, scale, cam_mode);
        app.start();
    } catch (const std::exception &e) {
        log_error("Main", std::string("Fatal: ") + e.what());
        return 1;
    }

    return 0;
}
