#include "fitter_app.h"

#include <algorithm>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "../utils/logs.h"

const int IMAGE_PADDING_PX          = 5;  // keeps the target image from touching the window edge
const int GRAPH_HISTORY_SIZE        = 100;
const int UPDATE_FPS_EVERY_N_FRAMES = 5;

FitterApp::FitterApp(int width, int height, const std::string &image_path, int splat_count)
    : AppBase(width, height, "Image Fitter", false)
    , image_path(image_path)
    , splat_count(splat_count)
{
    log_info("FitterApp",
        "Width=" + std::to_string(width) +
        " Height=" + std::to_string(height) +
        " Image=" + image_path +
        " SplatCount=" + std::to_string(splat_count)
    );
}

/* ===== ===== App overrides ===== ===== */

void FitterApp::onStart()
{
    fitter.init(width, height);
    fitter.loadTargetImage(image_path, width, height, IMAGE_PADDING_PX);
    fitter.randomInitGaussians(splat_count);
    // layers can only be wired after gaussians are initialized
    // as it needs to know the gaussian count for allocation
    fitter.initLayers();
}

void FitterApp::onFrame()
{
    fitter.step();
    displayFrame(fitter.getOutput());

    float current_loss = getLoss();

    static int frame_count = 0;
    frame_count++;
    if (frame_count % UPDATE_FPS_EVERY_N_FRAMES == 0
        && fitter.is_optimization_running)
    {
        loss_history.push_back(current_loss);
        iter_history.push_back(static_cast<float>(getIterCount()));
        if (loss_history.size() > GRAPH_HISTORY_SIZE) {
            loss_history.erase(loss_history.begin());
            iter_history.erase(iter_history.begin());
        }
    }

    // ImGui
    ImGui::SetNextWindowPos(ImVec2(2, 2), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(260, 300), ImGuiCond_Once);
    ImGui::Begin("Image Fitter");

    ImGui::Text("FPS: %.2f\t| Frametime: %.2f ms", getFPS(), getFrametime());
    ImGui::Text("Iteration: %d", getIterCount());
    ImGui::Text("Loss: %.8f", current_loss);

    ImGui::Separator();

    // Save PLY
    ImGui::InputText("##plypath", save_ply_path, sizeof(save_ply_path));
    ImGui::SameLine();
    if (ImGui::Button("Save PLY"))
        fitter.savePLY(save_ply_path);

    ImGui::Separator();

    // Pause / Continue toggle
    {
        // Determine color based on state
        ImVec4 running_color = fitter.is_optimization_running ? ImVec4(0.2f, 0.2f, 0.2f, 1.0f)
                                                            : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, running_color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(running_color.x+0.1f, running_color.y+0.1f, running_color.z+0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(running_color.x-0.1f, running_color.y-0.1f, running_color.z-0.1f, 1.0f));

        // Make the button a fixed width so label changes don't resize it
        float button_width = 60.0f;
        if (ImGui::Button("Run", ImVec2(button_width, 0))) {
            fitter.is_optimization_running = !fitter.is_optimization_running;
        }
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        ImGui::TextUnformatted(fitter.is_optimization_running ? "Running" : "Paused");
    }

    int history_size = static_cast<int>(loss_history.size());
    if (history_size > 0 && ImPlot::BeginPlot("##LossRolling", ImVec2(-1, ImGui::GetTextLineHeight() * 12))) {

        ImPlot::SetupAxes("Iteration", "Loss", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);

        // X axis
        float x_min = iter_history.front();
        float x_max = iter_history.back();
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Always);

        // Y axis
        float min_loss = *std::min_element(loss_history.begin(), loss_history.end());
        float max_loss = *std::max_element(loss_history.begin(), loss_history.end());
        if (min_loss == max_loss) max_loss += 1e-6f;
        float padding = (max_loss - min_loss) * 0.2f;
        ImPlot::SetupAxisLimits(ImAxis_Y1, min_loss - padding, max_loss + padding, ImPlotCond_Always);

        ImPlotSpec spec;
        spec.LineColor = ImVec4(0.9f, 0.9f, 0.4f, 1.0f);
        spec.LineWeight = 2.0f;
        ImPlot::PlotLine("Loss", iter_history.data(), loss_history.data(), history_size, spec);

        ImPlot::EndPlot();
    }

    ImGui::End();
}
