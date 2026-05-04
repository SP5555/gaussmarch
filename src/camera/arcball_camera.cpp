#include "arcball_camera.h"

#include <algorithm>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>

#include "../utils/ansi_colors.h"
#include "../utils/logs.h"

ArcballCamera::ArcballCamera(float aspect, float fovDegrees, float nearPlane, float farPlane)
    : aspect(aspect)
    , near_plane(nearPlane)
    , far_plane(farPlane)
    , fov(glm::radians(fovDegrees))
{
    log_info("ArcballCamera",
        "Controls:\n"
        "  Left Click + Drag         -> Orbit\n"
        "  Shift + Left Click + Drag -> Pan\n"
        "  Scroll                    -> Zoom",
        ANSI_CYAN ANSI_BOLD
    );

    updateViewSpaceVectors();
    updateMatrices();
}

/* ===== ===== Update ===== ===== */

bool ArcballCamera::update(const Input &input, float dt)
{
    // clamp large deltas from click jumps
    float mouseDelta_x =  glm::clamp(input.mouse_delta.x, -50.f, 50.f);
    float mouseDelta_y = -glm::clamp(input.mouse_delta.y, -50.f, 50.f);

    glm::vec3 offset   = position - target;
    glm::vec3 viewDir  = glm::normalize(-offset);
    float     distance = glm::length(offset);

    bool isDirty = false;

    // ===== orbit =====
    if (!input.isShiftDown() && input.mouse_left_held && (mouseDelta_x != 0.f || mouseDelta_y != 0.f))
    {
        glm::vec3 dir = glm::normalize(offset);

        float pitch = glm::asin(glm::clamp(dir.y, -1.f, 1.f));
        float yaw   = glm::atan(dir.x, dir.z);

        pitch -= mouseDelta_y * speed_rotate * dt;
        yaw   -= mouseDelta_x * speed_rotate * dt;

        pitch = glm::clamp(pitch, MIN_PITCH, MAX_PITCH);
        yaw   = glm::mod(yaw, glm::two_pi<float>());

        glm::vec3 newDir = {
            glm::sin(yaw) * glm::cos(pitch),
            glm::sin(pitch),
            glm::cos(yaw) * glm::cos(pitch)
        };

        position = target + newDir * distance;
        isDirty  = true;
    }

    // ===== pan =====
    if (input.isShiftDown() && input.mouse_left_held && (mouseDelta_x != 0.f || mouseDelta_y != 0.f))
    {
        glm::vec3 translation =
            right * (-mouseDelta_x * distance * speed_pan) +
            up    * (-mouseDelta_y * distance * speed_pan);

        position += translation;
        target   += translation;
        isDirty   = true;
    }

    // ===== zoom / ortho scale =====
    if (input.scroll_delta != 0.f)
    {
        if (ortho_mode)
        {
            ortho_half_h *= (1.f - input.scroll_delta * speed_zoom);
            ortho_half_h  = glm::max(ortho_half_h, 0.01f);
        }
        else
        {
            glm::vec3 translation = viewDir * (input.scroll_delta * distance * speed_zoom);
            position += translation;

            float newDist = glm::distance(position, target);
            if (newDist < zoom_min_dist)
                position = target + (-viewDir) * zoom_min_dist;
            if (newDist > zoom_max_dist)
                position = target + (-viewDir) * zoom_max_dist;
        }
        isDirty = true;
    }

    if (!isDirty) return false;
    updateViewSpaceVectors();
    updateMatrices();
    return true;
}

/* ===== ===== Resize ===== ===== */

void ArcballCamera::setAspect(float newAspect)
{
    aspect = newAspect;
    if (ortho_mode)
        p_matrix = glm::ortho(-ortho_half_h * aspect, ortho_half_h * aspect,
                              -ortho_half_h, ortho_half_h, near_plane, far_plane);
    else
        p_matrix = glm::perspective(fov, aspect, near_plane, far_plane);
}

void ArcballCamera::setOrthoMode(bool enable)
{
    if (enable && !ortho_mode)
        ortho_half_h = glm::tan(fov * 0.5f) * glm::distance(position, target);
    ortho_mode = enable;
    updateMatrices();
}

/* ===== ===== Helpers ===== ===== */

void ArcballCamera::updateViewSpaceVectors()
{
    forward = glm::normalize(target - position);
    right   = glm::normalize(glm::cross(forward, glm::vec3(0.f, 1.f, 0.f)));
    up      = glm::normalize(glm::cross(right, forward));
}

void ArcballCamera::updateMatrices()
{
    v_matrix = glm::lookAt(position, target, up);
    if (ortho_mode)
        p_matrix = glm::ortho(-ortho_half_h * aspect, ortho_half_h * aspect,
                              -ortho_half_h, ortho_half_h, near_plane, far_plane);
    else
        p_matrix = glm::perspective(fov, aspect, near_plane, far_plane);
}
