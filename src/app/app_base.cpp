#include "app_base.h"

#include <iostream>
#include <stdexcept>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include "../cuda/cuda_check.h"
#include "../io/image_io.h"
#include "../utils/ansi_colors.h"
#include "../utils/logs.h"

/* ===== ===== GL Boilerplate ===== ===== */

static const float QUAD[] = {
    // x,    y,    u,    v
    -1.f, -1.f,  0.f,  1.f,
     1.f, -1.f,  1.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f, -1.f,  0.f,  1.f,
     1.f,  1.f,  1.f,  0.f,
    -1.f,  1.f,  0.f,  0.f,
};

static const char *VS_SRC = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

static const char *FS_SRC = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
)glsl";

void AppBase::glfwErrorCallback(int error, const char *description)
{
    std::cerr << "[GLFW] Error " << error << ": " << description << "\n";
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        glDeleteShader(s);
        log_fatal("AppBase", std::string("Shader compile error: ") + log);
    }
    return s;
}

/* ===== ===== Lifecycle ===== ===== */

AppBase::AppBase(int width, int height, const std::string &title, bool resizable)
    : width(width), height(height), title(title), resizable(resizable)
{
    PRINT_BUILD_INFO();

    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, resizable ? GLFW_TRUE : GLFW_FALSE);

    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwSetWindowTitle(window, title.c_str());
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImPlot::CreateContext();

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    log_info("App", std::string("OpenGL ")
        + (const char*)glGetString(GL_VERSION), ANSI_MAGENTA);
    log_info("App", std::string("GL Renderer: ")
        + (const char*)glGetString(GL_RENDERER), ANSI_MAGENTA);
    log_info("App", std::string("CUDA Device: ")
        + deviceProp.name, ANSI_MAGENTA);

    initGL();

    glfwSetWindowUserPointer(window, this);
    if (resizable) {
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow *win, int w, int h) {
            auto *app = static_cast<AppBase *>(glfwGetWindowUserPointer(win));
            if (!app) return;
            // update PBO/texture sizes
            app->onResize(w, h);
            // invoke the subclass's onWindowResize()
            // so it can react to the new size if needed
            app->onWindowResize(w, h);
        });
    }

    Input::install(window, &input);
}

AppBase::~AppBase()
{
    if (d_pbo_resource)
    {
        CUDA_WARN(cudaGraphicsUnregisterResource(d_pbo_resource));
        d_pbo_resource = nullptr;
    }

    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
}

void AppBase::renderOneFrame()
{
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse || io.WantCaptureKeyboard)
        input.flush();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    onFrame();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    double current_time = glfwGetTime();
    double delta_time   = current_time - last_frametime;
    dt                  = static_cast<float>(delta_time);
    last_frametime      = current_time;
    time_since_update  += delta_time;
    frame_since_update++;

    if (time_since_update >= 0.1)
    {
        avg_fps = avg_fps * 0.4f + (frame_since_update / (float)time_since_update) * 0.6f;
        time_since_update  = 0.0;
        frame_since_update = 0;
    }

    glfwSwapBuffers(window);

    bool f12Pressed = (glfwGetKey(window, GLFW_KEY_F12) == GLFW_PRESS);
    if (f12Pressed && !f12_was_pressed)
        saveScreenshot();
    f12_was_pressed = f12Pressed;

    // exit on ESC
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    input.flush();
}

void AppBase::start()
{
    // On Windows, dragging the resize handle blocks glfwPollEvents() inside a
    // modal loop.  GLFW fires the window-refresh callback from within that loop,
    // so rendering here keeps frames alive during the resize drag.
    glfwSetWindowRefreshCallback(window, [](GLFWwindow *win) {
        auto *app = static_cast<AppBase *>(glfwGetWindowUserPointer(win));
        if (app) app->renderOneFrame();
    });

    onStart();
    last_frametime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        renderOneFrame();
    }
}

/* ===== ===== Private Init ===== ===== */

void AppBase::initGL()
{
    vao.allocate();
    vbo.allocate();
    texture.allocate();

    setupFullscreenQuad(vao, vbo);
    linkShaderProgram(shader_program, VS_SRC, FS_SRC);
    createTextureAndPBO(width, height);
}

void AppBase::resizePBO(int width, int height)
{
    pbo.allocate();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(pbo));
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(float), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&d_pbo_resource, static_cast<GLuint>(pbo),
                                                    cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "[AppBase] PBO registration failed, falling back to host copy: "
                  << cudaGetErrorString(err) << "\n";
        pbo.release();
        cuda_GL_interop_supported = false;
        h_pixels.resize(width * height * 3);
    }
}

bool AppBase::checkCudaGLInterop()
{
    unsigned int deviceCount = 0;
    int glDevices[4];
    // cudaGLGetDevices sets a sticky error when interop is unavailable;
    // check the return value and clear the error so it doesn't poison later CUDA_SYNC_CHECKs.
    cudaError_t err = cudaGLGetDevices(&deviceCount, glDevices, 4, cudaGLDeviceListAll);
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // consume the sticky error
        return false;
    }

    int cudaDevice;
    CUDA_CHECK(cudaGetDevice(&cudaDevice));

    for (unsigned int i = 0; i < deviceCount; ++i)
        if (glDevices[i] == cudaDevice)
            return true;

    return false;
}

void AppBase::linkShaderProgram(GLShaderProgram &program, const char *vs_src, const char *fs_src)
{
    assert(!program.isValid() && "Shader program already allocated");
    program.allocate();
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vs_src);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_src);
    glAttachShader(static_cast<GLuint>(program), vs);
    glAttachShader(static_cast<GLuint>(program), fs);
    glLinkProgram(static_cast<GLuint>(program));
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok;
    glGetProgramiv(static_cast<GLuint>(program), GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(static_cast<GLuint>(program), 512, nullptr, log);
        log_error("AppBase", std::string("Shader link error: ") + log);
    }
}

void AppBase::setupFullscreenQuad(GLVertexArray &vao, GLBuffer &vbo)
{
    glBindVertexArray(static_cast<GLuint>(vao));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(vbo));
    glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void AppBase::createTextureAndPBO(int width, int height)
{
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    cuda_GL_interop_supported = checkCudaGLInterop();
    if (!cuda_GL_interop_supported)
    {
        h_pixels.resize(width * height * 3);
        return;
    }

    resizePBO(width, height);
}

/* ===== ===== Resize ===== ===== */

void AppBase::onResize(int newWidth, int newHeight)
{
    if (newWidth <= 0 || newHeight <= 0)
        return;

    if (newWidth == width && newHeight == height)
        return;

    width  = newWidth;
    height = newHeight;

    // recreate PBO for new size
    if (pbo.isValid())
    {
        if (d_pbo_resource)
        {
            CUDA_CHECK(cudaGraphicsUnregisterResource(d_pbo_resource));
            d_pbo_resource = nullptr;
        }
        pbo.release();
    }

    if (cuda_GL_interop_supported)
        resizePBO(width, height);
    else
        h_pixels.resize(newWidth * newHeight * 3);

    // resize texture
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, newWidth, newHeight, 0, GL_RGB, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glViewport(0, 0, newWidth, newHeight);
}

/* ===== ===== Display ===== ===== */

void AppBase::displayFrame(const float *d_pixels)
{
    last_pixels = d_pixels;

    if (cuda_GL_interop_supported)
    {
        CUDA_CHECK(cudaGraphicsMapResources(1, &d_pbo_resource));
        float  *d_pbo    = nullptr;
        size_t  pbo_size = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &pbo_size, d_pbo_resource));
        CUDA_CHECK(cudaMemcpy(d_pbo, d_pixels, width * height * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &d_pbo_resource));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(pbo));
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(h_pixels.data(), d_pixels,
                              width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, h_pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(static_cast<GLuint>(shader_program));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(texture));
    glBindVertexArray(static_cast<GLuint>(vao));
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

/* ===== ===== Screenshot ===== ===== */
void AppBase::saveScreenshot()
{
    if (!last_pixels) return;
    time_t t = time(nullptr);
    char buf[64];
    strftime(buf, sizeof(buf), "screenshot_%y%m%d_%H%M%S.png", localtime(&t));
    saveScreenshot(buf);
}

void AppBase::saveScreenshot(const char *path)
{
    saveScreenshot(std::string(path));
}

void AppBase::saveScreenshot(const std::string &path)
{
    ImageSaver::saveAsPNG(last_pixels, width, height, path);
    log_info("App", "Screenshot saved: " + path);
}

/* ===== ===== Window Title ===== ===== */
void AppBase::updateWindowTitle(const std::string &t)
{
    title = t;
    glfwSetWindowTitle(window, title.c_str());
}