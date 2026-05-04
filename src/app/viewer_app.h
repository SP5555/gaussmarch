#pragma once
#include <string>
#include <memory>
#include <vector>
#include "app_base.h"
#include "../pipelines/splat_renderer.h"
#include "../camera/camera.h"

enum class CameraMode { Arcball, Fly };

/**
 * @brief App for viewing a 3D Gaussian scene from a PLY file.
 *
 * Supports window resizing, orbit/pan/zoom camera controls.
 */
class ViewerApp : public AppBase
{
public:
    ViewerApp(const std::string &ply_path, float scene_scale = 1.f,
              CameraMode camera_mode = CameraMode::Fly);

protected:
    void onStart()  override;
    void onFrame() override;
    void onWindowResize(int newWidth, int newHeight) override;

private:
    void loadScene(const std::string &path);

    std::string    ply_path;
    float          scene_scale;
    CameraMode     camera_mode;
    SplatRenderer  renderer;

    std::unique_ptr<Camera> camera;

    // ImGui
    std::vector<float> fps_history;
    std::vector<float> x_axis;
    int         active_sh_degree = 0;
    bool        ortho_mode       = false;
    int         frame_count      = 0;
    std::string load_error;
};
