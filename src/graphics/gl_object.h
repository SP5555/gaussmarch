#pragma once

#include <utility>
#include <glad/glad.h>

/* ===== ===== GL RAII Wrappers ===== ===== */

template<typename Derived>
struct GLObject
{
    GLuint id = 0;

    GLObject() = default;
    ~GLObject() { if (id) static_cast<Derived*>(this)->release(); }

    GLObject(const GLObject &)            = delete;
    GLObject &operator=(const GLObject &) = delete;

    GLObject(GLObject &&other) noexcept : id(other.id) { other.id = 0; }
    GLObject &operator=(GLObject &&other) noexcept
    {
        if (this != &other)
        {
            if (id) static_cast<Derived*>(this)->release();
            id = other.id;
            other.id = 0;
        }
        return *this;
    }

    // true explicit conversion
    explicit operator GLuint() const { return id; }
    // explicit-ish getter
    GLuint get() const { return id; }
    // another explicit-ish if * syntax is prefered
    GLuint& operator*() { return id; }
    const GLuint& operator*() const { return id; }

    bool isValid() const { return id != 0; }
};

struct GLBuffer : GLObject<GLBuffer>
{
    GLBuffer() = default;
    void allocate() { glGenBuffers(1, &id); }
    void release() { glDeleteBuffers(1, &id); id = 0; }
};

struct GLTexture : GLObject<GLTexture>
{
    GLTexture() = default;
    void allocate() { glGenTextures(1, &id); }
    void release() { glDeleteTextures(1, &id); id = 0; }
};

struct GLVertexArray : GLObject<GLVertexArray>
{
    GLVertexArray() = default;
    void allocate() { glGenVertexArrays(1, &id); }
    void release() { glDeleteVertexArrays(1, &id); id = 0; }
};

struct GLShaderProgram : GLObject<GLShaderProgram>
{
    GLShaderProgram() = default;
    void allocate() { id = glCreateProgram(); }
    void release() { glDeleteProgram(id); id = 0; }
};