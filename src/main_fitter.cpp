#include <iostream>
#include "cxxopts.hpp"
#include "app/fitter_app.h"
#include "utils/logs.h"

int main(int argc, char *argv[])
{
    cxxopts::Options options("fitter", "Image Fitter");

    options.add_options()
        ("w,width",       "Window width",              cxxopts::value<int>()->default_value("1280"))
        ("h,height",      "Window height",             cxxopts::value<int>()->default_value("720"))
        ("i,image",       "Path to target image",      cxxopts::value<std::string>()->default_value("data/img/torii_moon.jpg"))
        ("s,splat-count", "Number of starting splats", cxxopts::value<int>()->default_value("60000"))
        ("help",          "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int width              = result["width"].as<int>();
    int height             = result["height"].as<int>();
    std::string image_path = result["image"].as<std::string>();
    int splat_count        = result["splat-count"].as<int>();

    /* ===== Argument Validation ===== */
    if ((result.count("width") && !result.count("height")) ||
        (!result.count("width") && result.count("height"))) {
        log_error("Main", "Error: --width and --height must be specified together.");
        return 1;
    }

    if (width <= 0 || height <= 0) {
        log_error("Main", "Error: width and height must be positive integers.");
        return 1;
    }

    if (!result.count("width") && !result.count("height")) {
        char buf[128];
        snprintf(buf, sizeof(buf), "No width/height specified, defaulting to %dx%d", width, height);
        log_info("Main", buf);
    }

    if (splat_count <= 0) {
        log_error("Main", "Error: splat count must be positive.");
        return 1;
    }

    try {
        FitterApp app(width, height, image_path, splat_count);
        app.start();
    }
    catch (const std::exception &e)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Failed to start application: %s", e.what());
        log_error("Main", buf);
        return 1;
    }

    return 0;
}