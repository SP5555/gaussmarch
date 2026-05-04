#include "fly_camera.h"

#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "../utils/ansi_colors.h"
#include "../utils/logs.h"

FlyCamera::FlyCamera(float aspect, float fovDegrees, float nearPlane, float farPlane)
    : aspect(aspect)
    , near_plane(nearPlane)
    , far_plane(farPlane)
    , fov(glm::radians(fovDegrees))
{
    log_info("FlyCamera",
        "Controls:\n"
        "  W/A/S/D           -> Move/Strafe\n"
        "  R/F               -> Move up/down\n"
        "  Q/E               -> Roll left/right\n"
        "  Left Click + Drag -> Look around (yaw/pitch)\n"
        "  Shift             -> Fast move\n"
        "  Ctrl              -> Slow move",
        ANSI_CYAN ANSI_BOLD
    );

    updateViewSpaceVectors();
    updateMatrices();
}

/* ===== ===== Update ===== ===== */

bool FlyCamera::update(const Input &input, float dt)
{
    // clamp large deltas from click jumps
    float mouse_delta_x =  glm::clamp(input.mouse_delta.x, -50.f, 50.f);
    float mouse_delta_y = -glm::clamp(input.mouse_delta.y, -50.f, 50.f);

    float speed_mult = 1.f;
    if (input.isShiftDown())
        speed_mult *= speed_mult_shift;
    if (input.isCtrlDown())
        speed_mult *= speed_mult_ctrl;
    
    float move_delta = speed_move * speed_mult * dt;
    float roll_delta = speed_roll * speed_mult * dt;
    float look_delta = speed_look * speed_mult * dt;

    bool is_dirty = false;

    // WASD forward/strafe on the local plane
    if (input.isKeyDown(GLFW_KEY_W)) { position += plane_forward * move_delta; is_dirty = true; }
    if (input.isKeyDown(GLFW_KEY_S)) { position -= plane_forward * move_delta; is_dirty = true; }
    if (input.isKeyDown(GLFW_KEY_A)) { position -= plane_right   * move_delta; is_dirty = true; }
    if (input.isKeyDown(GLFW_KEY_D)) { position += plane_right   * move_delta; is_dirty = true; }

    // RF up/down
    if (input.isKeyDown(GLFW_KEY_R)) { position += plane_up      * move_delta; is_dirty = true; }
    if (input.isKeyDown(GLFW_KEY_F)) { position -= plane_up      * move_delta; is_dirty = true; }

    // QE roll
    if (input.isKeyDown(GLFW_KEY_Q)) {
        orientation = orientation * glm::angleAxis(-roll_delta, glm::vec3(0.f, 0.f, -1.f));
        is_dirty = true;
    }
    if (input.isKeyDown(GLFW_KEY_E)) {
        orientation = orientation * glm::angleAxis( roll_delta, glm::vec3(0.f, 0.f, -1.f));
        is_dirty = true;
    }

    // mouse look (yaw/pitch)
    if (input.mouse_left_held)
    {
        if (mouse_delta_x != 0.f) {
            orientation = orientation * glm::angleAxis(
                -mouse_delta_x * look_delta,
                glm::vec3(0.f, 1.f, 0.f) // local up
            );
            is_dirty = true;
        }
        if (mouse_delta_y != 0.f) {
            pitch += mouse_delta_y * look_delta;
            pitch = glm::clamp(pitch, MIN_PITCH, MAX_PITCH);
            is_dirty = true;
        }
    }

    if (!is_dirty) return false;
    updateViewSpaceVectors();
    updateMatrices();
    return true;
}

/* ===== ===== Resize ===== ===== */

void FlyCamera::setAspect(float newAspect)
{
    aspect   = newAspect;
    p_matrix = glm::perspective(fov, aspect, near_plane, far_plane);
}

/* ===== ===== Helpers ===== ===== */

void FlyCamera::updateViewSpaceVectors()
{
    orientation = glm::normalize(orientation);

    plane_forward = orientation * glm::vec3( 0.f,  0.f, -1.f);
    plane_right   = orientation * glm::vec3( 1.f,  0.f,  0.f);
    plane_up      = orientation * glm::vec3( 0.f,  1.f,  0.f);
}

void FlyCamera::updateMatrices()
{
    glm::vec3 viewDir = glm::normalize(
        plane_forward * glm::cos(pitch) + plane_up * glm::sin(pitch)
    );

    glm::vec3 viewUp = glm::normalize(glm::cross(plane_right, viewDir));

    v_matrix = glm::lookAt(position, position + viewDir, viewUp);
    p_matrix = glm::perspective(fov, aspect, near_plane, far_plane);
}