#pragma once

#include <unordered_set>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Input
{
public:
    // mouse
    glm::vec2 mouse_pos   = {0.f, 0.f};
    glm::vec2 mouse_delta = {0.f, 0.f};
    bool  mouse_left_held  = false;
    bool  mouse_right_held = false;
    float scroll_delta       = 0.f;

    // key query
    bool isKeyDown(int glfwKey) const { return keys_down.count(glfwKey) > 0; }

    bool isShiftDown() const { return isKeyDown(GLFW_KEY_LEFT_SHIFT) || 
                                      isKeyDown(GLFW_KEY_RIGHT_SHIFT); }
    bool isCtrlDown()  const { return isKeyDown(GLFW_KEY_LEFT_CONTROL) || 
                                      isKeyDown(GLFW_KEY_RIGHT_CONTROL); }

    void flush();

    static void install(GLFWwindow *window, Input *input);

private:
    std::unordered_set<int> keys_down;

    glm::vec2 last_mouse_pos = {0.f, 0.f};
    bool first_mouse = true;

    // previous callbacks for chaining if there are multiple
    // entities that consume input from the same window
    GLFWmousebuttonfun prev_mouse_button_cb = nullptr;
    GLFWcursorposfun   prev_mouse_move_cb   = nullptr;
    GLFWscrollfun      prev_mouse_scroll_cb = nullptr;
    GLFWkeyfun         prev_key_cb          = nullptr;

    static void cbMouseButton(GLFWwindow *window, int button, int action, int mods);
    static void cbMouseMove  (GLFWwindow *window, double xpos, double ypos);
    static void cbMouseScroll(GLFWwindow *window, double xoffset, double yoffset);
    static void cbKey        (GLFWwindow *window, int key, int scancode, int action, int mods);
};