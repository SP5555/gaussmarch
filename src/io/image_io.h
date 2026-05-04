#pragma once
#include <string>
#include <vector>

struct LoadedImage
{
    // RGB [0, 1]
    std::vector<float> pixels;

    // dimensions
    int canvas_w = 0;
    int canvas_h = 0;
    int src_w = 0;
    int src_h = 0;
};

/**
 * @brief loads PNG/JPG, resizes to fit target dimensions while preserving aspect ratio.
 *
 * The pixels outside the original image area are filled with black.
 */
class ImageLoader
{
public:
    static LoadedImage load(
        const std::string &path,
        int target_w,
        int target_h,
        int padding = 0
    );
};

/**
 * @brief saves a float RGB buffer to a PNG file on disk.
 */
class ImageSaver
{
public:
    static void saveAsPNG(const float *d_pixels, int width, int height, const std::string &path);
};
