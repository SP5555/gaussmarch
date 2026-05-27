#pragma once
#include <string>
#include <memory>
#include <vector>

#include "app_base.h"
#include "../pipelines/gaussian_renderer.h"
#include "../camera/camera.h"
#include "tfn/widget.h"

enum class CameraMode { Arcball, Fly };

class GaussmarchApp : public AppBase
{
public:
    GaussmarchApp(const std::string &ply_path = "",
                  float scene_scale = 1.f,
                  CameraMode camera_mode = CameraMode::Arcball);

protected:
    void onStart()  override;
    void onFrame()  override;
    void onWindowResize(int w, int h) override;

private:
    void loadFile(const std::string &path);
    void drawUI();

    std::string        ply_path;
    float              scene_scale;
    CameraMode         camera_mode;
    GaussianRenderer   renderer;

    std::unique_ptr<Camera> camera;
    std::unique_ptr<tfn::TransferFunctionWidget> tf_widget;

    std::vector<float> fps_history;
    std::vector<float> fps_x;
    static constexpr int FPS_HISTORY = 100;

    std::string load_error;
    int frame_count = 0;
};
