#pragma once

#include <string>
#include <vector>

#include <glad/glad.h>       // must precede any GL headers
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // must follow glad (pulls in GL/gl.h)

#include "../graphics/gl_object.h"
#include "../input/input.h"

/**
 * @brief Base class for all apps.
 * 
 * Handles GLFW/GLAD init, window creation, CUDA device info, the main loop,
 * and display (fullscreen quad + PBO/host-copy paths).
 * 
 * Subclasses implement onStart(), onFrame(),
 * and optionally onInput() and onWindowResize().
 * 
 * Subclasses should call displayFrame(pixels) at the end of onFrame()
 * to push pixels to screen.
 */
class AppBase
{
public:
    virtual ~AppBase();

    void start();

protected:
    // derived class must call this as: AppBase(w, h, title, resizable)
    AppBase(int width, int height, const std::string &title, bool resizable);

    // override in subclass
    virtual void onStart() = 0; // called once before the loop
    virtual void onFrame() = 0; // called every frame

    // override when subclass needs sizes of resized framebuffer
    virtual void onWindowResize(int width, int height) {};

    // call at the end of onFrame() with CUDA device pixel buffer [H*W*3]
    // to display it on the screen.
    void displayFrame(const float *d_pixels);

    // screenshot utility
    void saveScreenshot(); // default path with timestamp
    void saveScreenshot(const char *path);
    void saveScreenshot(const std::string &path);

    // window title update utility
    void updateWindowTitle(const std::string &t);

    // expose states for ImGui to display
    float getFPS() const { return avg_fps; }
    float getFrametime() const { return 1000.f / avg_fps; } // in ms

    // window / input state
    GLFWwindow *window = nullptr;
    Input input;
    int width  = 0;
    int height = 0;
    bool resizable = false;

    // loop state
    float dt = 0.f;

private:
    void initGL();
    bool checkCudaGLInterop();
    void resizePBO(int width, int height);
    void linkShaderProgram(GLShaderProgram &program, const char *vs_src, const char *fs_src);
    void setupFullscreenQuad(GLVertexArray &vao, GLBuffer &vbo);
    void createTextureAndPBO(int width, int height);

    // used to update PBO/texture on window resize
    void onResize(int newWidth, int newHeight);

    void renderOneFrame();

    static void glfwErrorCallback(int error, const char *description);

    /* ---- GL objects ---- */
    GLShaderProgram shader_program;
    GLTexture       texture;
    GLVertexArray   vao;
    GLBuffer        vbo;
    GLBuffer        pbo;

    /* ---- CUDA/GL interop ---- */
    cudaGraphicsResource *d_pbo_resource            = nullptr;
    bool                  cuda_GL_interop_supported = false;
    std::vector<float>    h_pixels;

    /* ---- render state ---- */
    const float *last_pixels = nullptr;

    /* ---- screenshot state ---- */
    bool f12_was_pressed = false;

    /* ---- loop state ---- */
    double last_frametime     = 0.0;
    double time_since_update  = 0.0;
    int    frame_since_update = 0;
    float  avg_fps            = 0.f;
    std::string title;
};