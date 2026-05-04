#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"
#include "../input/input.h"

/**
 * @brief Orbit camera with pan and zoom.
 * 
 * Controls:
 *   Left drag         -> orbit (yaw / pitch around target)
 *   Shift + left drag -> pan (translate position and target together)
 *   Scroll            -> zoom (dolly along view direction)
 * 
 * Call update() every frame with mouse/scroll deltas.
 * Call getViewMatrix() / getProjectionMatrix() to feed the renderer.
 * Call setAspect() on window resize.
 */
class ArcballCamera : public Camera
{
public:
    ArcballCamera(float aspect, float fovDegrees = 40.f,
           float nearPlane = 0.1f, float farPlane = 100.f);

    // call every frame, returns true if matrices changed
    bool update(const Input &input, float dt);

    // call on window resize
    void setAspect(float aspect);

    // toggle orthographic projection; initial ortho scale is matched to current persp view
    void setOrthoMode(bool enable);

    const glm::vec3 &getPosition()         const { return position; }
    const glm::mat4 &getViewMatrix()       const { return v_matrix; }
    const glm::mat4 &getProjectionMatrix() const { return p_matrix; }

    // ---- config ----
    float speed_rotate  = 0.2f;
    float speed_pan     = 0.005f;
    float speed_zoom    = 0.1f;
    float zoom_min_dist = 0.5f;
    float zoom_max_dist = 50.f;

private:
    void updateViewSpaceVectors();
    void updateMatrices();

    // ---- state ----
    glm::vec3 position = {0.f, 0.f, 3.f};
    glm::vec3 target   = {0.f, 0.f, 0.f};

    glm::vec3 forward  = {0.f, 0.f, -1.f};
    glm::vec3 right    = {1.f, 0.f,  0.f};
    glm::vec3 up       = {0.f, 1.f,  0.f};

    // ---- projection params ----
    float fov = glm::radians(60.f);
    float aspect;
    float near_plane;
    float far_plane;
    bool  ortho_mode   = false;
    float ortho_half_h = 1.f; // half-height of ortho frustum in world units

    // ---- cached matrices ----
    glm::mat4 v_matrix = glm::mat4(1.f);
    glm::mat4 p_matrix = glm::mat4(1.f);

    static constexpr float MIN_PITCH = -glm::half_pi<float>() + 0.01f;
    static constexpr float MAX_PITCH =  glm::half_pi<float>() - 0.01f;
};
