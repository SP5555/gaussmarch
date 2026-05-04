#include "input.h"

#include <unordered_map>

static std::unordered_map<GLFWwindow*, Input*> s_inputs;

void Input::flush()
{
    mouse_delta  = {0.f, 0.f};
    scroll_delta = 0.f;
}

void Input::install(GLFWwindow *window, Input *input)
{
    s_inputs[window] = input;
    input->prev_mouse_button_cb = glfwSetMouseButtonCallback(window, cbMouseButton);
    input->prev_mouse_move_cb   = glfwSetCursorPosCallback  (window, cbMouseMove);
    input->prev_mouse_scroll_cb = glfwSetScrollCallback     (window, cbMouseScroll);
    input->prev_key_cb          = glfwSetKeyCallback        (window, cbKey);
}

void Input::cbMouseButton(GLFWwindow *window, int button, int action, int mods)
{
    Input *input   = s_inputs[window];
    bool   pressed = (action == GLFW_PRESS);
    if      (button == GLFW_MOUSE_BUTTON_LEFT)  input->mouse_left_held  = pressed;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) input->mouse_right_held = pressed;

    if (input->prev_mouse_button_cb)
        input->prev_mouse_button_cb(window, button, action, mods);
}

void Input::cbMouseMove(GLFWwindow *window, double xpos, double ypos)
{
    Input    *input  = s_inputs[window];
    glm::vec2 newPos = {(float)xpos, (float)ypos};

    if (input->first_mouse)
    {
        input->last_mouse_pos = newPos;
        input->first_mouse   = false;
    }

    input->mouse_delta   += newPos - input->last_mouse_pos;
    input->last_mouse_pos = newPos;
    input->mouse_pos      = newPos;

    if (input->prev_mouse_move_cb)
        input->prev_mouse_move_cb(window, xpos, ypos);
}

void Input::cbMouseScroll(GLFWwindow *window, double xoffset, double yoffset)
{
    Input *input         = s_inputs[window];
    input->scroll_delta += (float)yoffset;

    if (input->prev_mouse_scroll_cb)
        input->prev_mouse_scroll_cb(window, xoffset, yoffset);
}

void Input::cbKey(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Input *input = s_inputs[window];
    if (action == GLFW_PRESS)
        input->keys_down.insert(key);
    else if (action == GLFW_RELEASE)
        input->keys_down.erase(key);

    if (input->prev_key_cb)
        input->prev_key_cb(window, key, scancode, action, mods);
}