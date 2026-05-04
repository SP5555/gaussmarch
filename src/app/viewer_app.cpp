#include "viewer_app.h"

#include <algorithm>
#include <iostream>
#include <numeric>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "tinyfiledialogs.h"

#include <cuda_runtime.h>

#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"
#include "../utils/logs.h"

const int START_WIDTH = 1280;
const int START_HEIGHT = 720;

// ImPlot
const int GRAPH_HISTORY_SIZE = 100;
const int UPDATE_FPS_EVERY_N_FRAMES = 2;

ViewerApp::ViewerApp(const std::string &ply_path, float scene_scale, CameraMode camera_mode)
    : AppBase(START_WIDTH, START_HEIGHT, "Splat viewer", true)
    , ply_path(ply_path)
    , scene_scale(scene_scale)
    , camera_mode(camera_mode)
{
    float aspect = (float)START_WIDTH / START_HEIGHT;
    if (camera_mode == CameraMode::Arcball)
        camera = std::make_unique<ArcballCamera>(aspect);
    else if (camera_mode == CameraMode::Fly)
        camera = std::make_unique<FlyCamera>(aspect);

    log_info("ViewerApp",
        "PLY=" + ply_path +
        " Scale=" + std::to_string(scene_scale) +
        " Camera=" + (camera_mode == CameraMode::Arcball ? "Arcball" : "Fly")
    );

    x_axis.resize(GRAPH_HISTORY_SIZE);
    std::iota(x_axis.begin(), x_axis.end(), 0.f);
}

/* ===== ===== Helpers ===== ===== */

void ViewerApp::loadScene(const std::string &path)
{
    load_error.clear();
    try {
        renderer.reloadPLY(path, scene_scale);
        active_sh_degree = renderer.getMaxSHDegree();
    } catch (const std::exception &e) {
        load_error = e.what();
        log_error("ViewerApp", "Failed to load scene: " + load_error);
        cudaGetLastError(); // clear any sticky CUDA error state
    }
}

/* ===== ===== App overrides ===== ===== */

void ViewerApp::onStart()
{
    renderer.init(width, height);

    if (!ply_path.empty())
        loadScene(ply_path);
}

void ViewerApp::onFrame()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && !io.WantCaptureKeyboard)
        camera->update(input, dt);

    if (renderer.isLoaded()) {
        renderer.render(camera->getViewMatrix(), camera->getProjectionMatrix(), camera->getPosition());
        displayFrame(renderer.getOutput());
    }

    frame_count++;
    if (renderer.isLoaded() && frame_count % UPDATE_FPS_EVERY_N_FRAMES == 0) {
        fps_history.push_back(getFPS());
        if (fps_history.size() > GRAPH_HISTORY_SIZE)
            fps_history.erase(fps_history.begin());
    }

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(260, 260), ImGuiCond_Once);
    ImGui::Begin("Splat Viewer");

    if (ImGui::Button("Open PLY...")) {
        const char *filter_patterns[] = {"*.ply"};
        const char *result = tinyfd_openFileDialog("Open PLY Scene", "", 1, filter_patterns, "PLY files", 0);
        if (result) {
            ply_path = result;
            loadScene(ply_path);
            fps_history.clear();
        }
    }

    if (!load_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 0.4f, 0.4f, 1.f));
        ImGui::TextWrapped("Error: %s", load_error.c_str());
        ImGui::PopStyleColor();
    }

    if (renderer.isLoaded()) {
        ImGui::Text("Visible Splats: %d", renderer.getVisibleCount());
        auto pos = camera->getPosition();
        ImGui::Text("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("FPS: %.2f\t| Frametime: %.2f ms", getFPS(), getFrametime());

        if (camera_mode == CameraMode::Arcball) {
            if (ImGui::Checkbox("Orthographic", &ortho_mode))
                camera->setOrthoMode(ortho_mode);
        }

        int max_sh = renderer.getMaxSHDegree();
        if (max_sh > 0) {
            const char *sh_items[] = {"0 (DC only)", "1", "2", "3"};
            if (ImGui::Combo("SH Degree", &active_sh_degree, sh_items, max_sh + 1))
                renderer.setActiveSHDegree(active_sh_degree);
        }

        int history_size = static_cast<int>(fps_history.size());
        if (history_size > 0 && ImPlot::BeginPlot("##FPSRolling", ImVec2(-1, ImGui::GetTextLineHeight() * 8))) {

            ImPlot::SetupAxes(nullptr, "FPS", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);

            ImPlot::SetupAxisLimits(ImAxis_X1, 0, GRAPH_HISTORY_SIZE, ImPlotCond_Always);

            float max_fps = *std::max_element(fps_history.begin(), fps_history.end());
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_fps + 20, ImPlotCond_Always);

            ImPlotSpec spec;
            spec.LineColor = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
            spec.LineWeight = 2.0f;
            ImPlot::PlotLine("FPS", x_axis.data(), fps_history.data(), history_size, spec);

            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

void ViewerApp::onWindowResize(int newWidth, int newHeight)
{
    camera->setAspect((float)newWidth / newHeight);
    renderer.resize(newWidth, newHeight);
}
