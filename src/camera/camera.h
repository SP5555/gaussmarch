#pragma once
#include <glm/glm.hpp>
#include "../input/input.h"

class Camera
{
public:
    virtual ~Camera() = default;

    virtual bool update(const Input &input, float dt) = 0;
    virtual void setAspect(float aspect)              = 0;
    virtual void setOrthoMode(bool /*enable*/)         {}

    virtual const glm::vec3 &getPosition()         const = 0;
    virtual const glm::mat4 &getViewMatrix()       const = 0;
    virtual const glm::mat4 &getProjectionMatrix() const = 0;
};