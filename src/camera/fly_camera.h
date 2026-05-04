#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "camera.h"
#include "../input/input.h"

/**
 * @brief Free-fly camera with full 6DOF movement.
 * 
 * The camera maintains a local coordinate frame via a quaternion.
 * Movement is relative to the camera's current orientation.
 * 
 * Controls:
 *   W/S             -> move forward/backward on local plane
 *   A/D             -> strafe left/right on local plane
 *   R/F             -> move up/down along local up axis
 *   Q/E             -> roll left/right (rotates the entire local plane)
 *   Mouse drag      -> look (yaw/pitch)
 *   Scroll          -> move speed multiplier
 * 
 * Call update() every frame.
 * Call getViewMatrix() / getProjectionMatrix() to feed the renderer.
 * Call setAspect() on window resize.
 */
class FlyCamera : public Camera
{
public:
    FlyCamera(float aspect, float fovDegrees = 40.f,
            float nearPlane = 0.1f, float farPlane = 100.f);
    
    // call every frame, returns true if matrices changed
    bool update(const Input &input, float dt);

    // call on window resize
    void setAspect(float aspect);

    const glm::vec3 &getPosition()         const { return position; }
    const glm::mat4 &getViewMatrix()       const { return v_matrix; }
    const glm::mat4 &getProjectionMatrix() const { return p_matrix; }

    // ---- config ----
    float speed_move       = 5.f;
    float speed_look       = 0.2f;
    float speed_roll       = 0.2f;
    float speed_mult_shift = 4.f;
    float speed_mult_ctrl  = 0.2f;

private:
    void updateViewSpaceVectors();
    void updateMatrices();

    // ---- plane ----
    glm::quat orientation = glm::quat(1.f, 0.f, 0.f, 0.f);

    // ---- camera ----
    glm::vec3 position = {0.f, 0.f, 3.f};
    float pitch = 0.f;

    glm::vec3 plane_forward  = {0.f, 0.f, -1.f};
    glm::vec3 plane_right    = {1.f, 0.f,  0.f};
    glm::vec3 plane_up       = {0.f, 1.f,  0.f};

    // ---- projection params ----
    float fov = glm::radians(60.f);
    float aspect;
    float near_plane;
    float far_plane;

    // ---- cached matrices ----
    glm::mat4 v_matrix = glm::mat4(1.f);
    glm::mat4 p_matrix = glm::mat4(1.f);

    static constexpr float MIN_PITCH = -glm::half_pi<float>() + 0.01f;
    static constexpr float MAX_PITCH =  glm::half_pi<float>() - 0.01f;
};