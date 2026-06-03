#include <iostream>
#include <algorithm>
#include <string>

#include "cxxopts.hpp"
#include "app/volmarch_app.h"
#include "utils/logs.h"

int main(int argc, char *argv[])
{
    cxxopts::Options options("volmarch", "Raw volume renderer");
    options.add_options()
        ("s,scene",  "Path to raw volume file (e.g. vorts1_128x128x128_float32.raw)",
                     cxxopts::value<std::string>()->default_value(""))
        ("c,camera", "Camera mode: fly | arcball",
                     cxxopts::value<std::string>()->default_value("arcball"))
        ("help",     "Print usage");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << "\n";
        return 0;
    }

    std::string path    = result["scene"].as<std::string>();
    std::string cam_str = result["camera"].as<std::string>();
    std::transform(cam_str.begin(), cam_str.end(), cam_str.begin(), ::tolower);

    CameraMode cam_mode = CameraMode::Arcball;
    if (cam_str == "fly") cam_mode = CameraMode::Fly;
    else if (cam_str != "arcball") {
        log_error("Main", "Invalid camera mode. Valid options: fly, arcball.");
        return 1;
    }

    try {
        VolmarchApp app(path, cam_mode);
        app.start();
    } catch (const std::exception &e) {
        log_error("Main", std::string("Fatal: ") + e.what());
        return 1;
    }

    return 0;
}
