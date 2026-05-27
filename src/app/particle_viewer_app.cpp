#include "particle_viewer_app.h"

#include <numeric>
#include <algorithm>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "tinyfiledialogs.h"

#include "../camera/arcball_camera.h"
#include "../camera/fly_camera.h"
#include "../utils/logs.h"

static constexpr int START_W = 1280;
static constexpr int START_H = 720;

ParticleViewerApp::ParticleViewerApp(const std::string &path, float scale, CameraMode cam_mode)
    : AppBase(START_W, START_H, "gaussmarch -- Gaussian volume renderer", true)
    , ply_path(path)
    , scene_scale(scale)
    , camera_mode(cam_mode)
{
    float aspect = (float)START_W / START_H;
    if (camera_mode == CameraMode::Arcball)
        camera = std::make_unique<ArcballCamera>(aspect);
    else
        camera = std::make_unique<FlyCamera>(aspect);

    fps_x.resize(FPS_HISTORY);
    std::iota(fps_x.begin(), fps_x.end(), 0.f);
}

void ParticleViewerApp::loadFile(const std::string &path)
{
    load_error.clear();
    try {
        renderer.loadGaussians(path, scene_scale);
    } catch (const std::exception &e) {
        load_error = e.what();
        log_error("ParticleViewerApp", "Load failed: " + load_error);
    }
}

void ParticleViewerApp::onStart()
{
    renderer.init(optix_context, width, height);

    // Build TF widget after GL is initialized (it allocates a GL texture).
    tf_widget = std::make_unique<tfn::TransferFunctionWidget>(
        [this](const tfn::list3f &colors, const tfn::list2f &alphas, const tfn::vec2f &)
        {
            int n = (int)colors.size();
            if (n == 0) return;
            std::vector<float> rgba4(n * 4);
            for (int i = 0; i < n; ++i) {
                rgba4[i*4+0] = colors[i].x;
                rgba4[i*4+1] = colors[i].y;
                rgba4[i*4+2] = colors[i].z;
                rgba4[i*4+3] = alphas[i].y;
            }
            renderer.setColormap(rgba4.data(), n);
        });

    if (!ply_path.empty())
        loadFile(ply_path);
}

void ParticleViewerApp::onFrame()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && !io.WantCaptureKeyboard)
        camera->update(input, dt);

    if (renderer.isLoaded()) {
        renderer.render(
            camera->getViewMatrix(),
            camera->getProjectionMatrix(),
            camera->getPosition());
        displayFrame(renderer.getOutput());
    }

    ++frame_count;
    fps_history.push_back(getFPS());
    if ((int)fps_history.size() > FPS_HISTORY)
        fps_history.erase(fps_history.begin());

    drawUI();
}

void ParticleViewerApp::onWindowResize(int w, int h)
{
    camera->setAspect((float)w / h);
    renderer.resize(w, h);
}

void ParticleViewerApp::drawUI()
{
    // ===== Main control window =====
    ImGui::SetNextWindowPos({2, 2}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({280, 400}, ImGuiCond_Once);
    ImGui::Begin("gaussmarch");

    if (ImGui::Button("Open .ply...")) {
        const char *filters[] = {"*.ply"};
        const char *result = tinyfd_openFileDialog(
            "Open VEG PLY file", "", 1, filters, "VEG PLY files (*.ply)", 0);
        if (result) {
            ply_path = result;
            loadFile(ply_path);
            fps_history.clear();
        }
    }

    if (!load_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, {1.f, 0.4f, 0.4f, 1.f});
        ImGui::TextWrapped("Error: %s", load_error.c_str());
        ImGui::PopStyleColor();
    }

    ImGui::Separator();

    if (renderer.isLoaded()) {
        ImGui::Text("Gaussians: %d", renderer.getGaussianCount());
        {
            auto p = camera->getPosition();
            ImGui::Text("Camera: (%.2f, %.2f, %.2f)", p.x, p.y, p.z);
        }
        ImGui::Text("FPS: %.1f  |  %.2f ms", getFPS(), getFrametime());

        ImGui::Separator();

        // March parameters
        ImGui::Text("March");
        if (ImGui::SliderFloat("Step size", &renderer.step_size, 0.001f, 0.1f, "%.3f"))
            renderer.resetAccum();
        if (ImGui::SliderInt("Max depth", &renderer.max_depth, 64, 2048))
            renderer.resetAccum();

        ImGui::Separator();

        // Rendering
        ImGui::Text("Rendering");
        if (ImGui::Checkbox("Frame Accumulation", &renderer.accum_enabled))
            renderer.resetAccum();
        if (renderer.accum_enabled) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset"))
                renderer.resetAccum();
        }
        if (ImGui::Checkbox("Blue Noise", &renderer.blue_noise_enabled))
            renderer.resetAccum();

        ImGui::Separator();

        // Lighting
        ImGui::Text("Lighting");
        if (ImGui::Checkbox("Shadows", &renderer.shadows_enabled))
            renderer.resetAccum();

        ImGui::BeginDisabled(!renderer.shadows_enabled);
        {
            float az_deg = renderer.light_azimuth * 360.f;
            float el_deg = (renderer.light_elevation - 0.5f) * 180.f;   // -90=below, 0=horizon, 90=above
            if (ImGui::SliderFloat("Azimuth##deg",   &az_deg, 0.f, 360.f, "%.1f deg"))
                { renderer.light_azimuth   = az_deg / 360.f; renderer.resetAccum(); }
            if (ImGui::SliderFloat("Elevation##deg", &el_deg, -90.f, 90.f, "%.1f deg"))
                { renderer.light_elevation = el_deg / 180.f + 0.5f; renderer.resetAccum(); }
        }
        if (ImGui::SliderFloat("Ambient", &renderer.light_ambient, 0.f, 1.f, "%.2f"))
            renderer.resetAccum();
        ImGui::EndDisabled();

        ImGui::Separator();

        // FPS plot
        if (!fps_history.empty() &&
            ImPlot::BeginPlot("##fps", {-1, ImGui::GetTextLineHeight() * 6}))
        {
            ImPlot::SetupAxes(nullptr, "FPS",
                ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, FPS_HISTORY, ImPlotCond_Always);
            float mx = *std::max_element(fps_history.begin(), fps_history.end());
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, mx + 10, ImPlotCond_Always);
            ImPlotSpec fps_spec;
            fps_spec.LineColor  = {0.4f, 0.9f, 0.4f, 1.f};
            fps_spec.LineWeight = 1.5f;
            ImPlot::PlotLine("FPS", fps_x.data(), fps_history.data(),
                             (int)fps_history.size(), fps_spec);
            ImPlot::EndPlot();
        }
    }

    ImGui::End();

    if (!renderer.isLoaded()) return;

    // ===== Transfer function =====
    ImGui::SetNextWindowPos({2, 410}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({420, 360}, ImGuiCond_Once);
    ImGui::Begin("Transfer Function");
    tf_widget->build_gui();
    ImGui::End();
    tf_widget->render();
}
