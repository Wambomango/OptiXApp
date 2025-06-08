#include "window.hpp"


Window::Window(int width, int height, std::string titel) : context(nullptr, true, true), width(width), height(height), titel(titel)
{
    context.Resize(width, height);
    context.Activate();

    screenquad_vao = std::make_unique<OpenGL::VertexArray>();
    screenquad_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("screenquad.vert"), SHADER_DIR + std::string("screenquad.frag"));

    glfwSetWindowUserPointer(context.window, this);
    glfwSetScrollCallback(context.window, &Window::ScrollDispatcher);
    glfwSetCursorPosCallback(context.window, &Window::MouseMoveDispatcher);
    glfwSetMouseButtonCallback(context.window, &Window::MouseButtonDispatcher);
    glfwSetKeyCallback(context.window, &Window::KeyDispatcher);

    GL_CREATE_TEXTURE_2D(output_texture, GL_RGBA8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_OUTPUT, output_texture);
}

Window::~Window()
{
    screenquad_vao.release();
    screenquad_program.release();
    glDeleteTextures(1, &output_texture);
}

float Window::GetTime()
{
    return static_cast<float>(glfwGetTime());
}

int Window::AddScrollCallback(std::function<void(double, double)> callback)
{
    int key = rand();
    while(scroll_callbacks.find(key) != scroll_callbacks.end())
    {
        key = rand();
    }
    scroll_callbacks[key] = callback;
    return key;
}

int Window::AddMouseMoveCallback(std::function<void(double, double)> callback)
{
    int key = rand();
    while(mouse_move_callbacks.find(key) != mouse_move_callbacks.end())
    {
        key = rand();
    }
    mouse_move_callbacks[key] = callback;
    return key;
}

int Window::AddMouseButtonCallback(std::function<void(int, int, int)> callback)
{
    int key = rand();
    while(mouse_button_callbacks.find(key) != mouse_button_callbacks.end())
    {
        key = rand();
    }
    mouse_button_callbacks[key] = callback;
    return key;
}

int Window::AddKeyCallback(std::function<void(int, int, int, int)> callback)
{
    int key = rand();
    while(key_callbacks.find(key) != key_callbacks.end())
    {
        key = rand();
    }
    key_callbacks[key] = callback;

    SPDLOG_WARN("Adding key callback with key {}", key);
    return key; 
}

bool Window::RemoveScrollCallback(int key)
{
    if(scroll_callbacks.find(key) == scroll_callbacks.end())
    {
        return false; 
    }

    scroll_callbacks.erase(key);
    return true;
}

bool Window::RemoveMouseMoveCallback(int key)
{
    if(mouse_move_callbacks.find(key) == mouse_move_callbacks.end())
    {
        return false;
    }

    mouse_move_callbacks.erase(key);
    return true;
}

bool Window::RemoveMouseButtonCallback(int key)
{
    if(mouse_button_callbacks.find(key) == mouse_button_callbacks.end())
    {
        return false;
    }
    mouse_button_callbacks.erase(key);
    return true;
}

bool Window::RemoveKeyCallback(int key)
{
    if(key_callbacks.find(key) == key_callbacks.end())
    {
        return false;
    }
    key_callbacks.erase(key);
    return true;
}

GLuint Window::GetTexture()
{
    return output_texture;
}

int Window::GetWidth()
{
    return width;
}

int Window::GetHeight()
{
    return height;
}
        
void Window::Resize(int new_width, int new_height)
{
    width = new_width;
    height = new_height;
    context.Resize(width, height);
}

bool Window::ShouldClose()
{
    return glfwWindowShouldClose(context.window);
}

void Window::Render()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);

    screenquad_program->Use();
    screenquad_vao->Bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Window::SwapBuffers()
{
    glfwSwapBuffers(context.window);
}

void Window::PollEvents()
{
    glfwPollEvents();
}

void Window::ScrollDispatcher(GLFWwindow* window, double xoffset, double yoffset)
{
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    if (self->scroll_callbacks.empty()) return;

    for (const auto& [key, callback] : self->scroll_callbacks)
    {
        callback(xoffset, yoffset);
    }
}

void Window::MouseMoveDispatcher(GLFWwindow* window, double xpos, double ypos)
{
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    if (self->mouse_move_callbacks.empty()) return;

    for (const auto& [key, callback] : self->mouse_move_callbacks)
    {
        callback(xpos, ypos);
    }
}

void Window::MouseButtonDispatcher(GLFWwindow* window, int button, int action, int mods)
{
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (!self) return;
    if (self->mouse_button_callbacks.empty()) return;

    for (const auto& [key, callback] : self->mouse_button_callbacks)
    {
        callback(button, action, mods);
    }
}

void Window::KeyDispatcher(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Window* self = static_cast<Window*>(glfwGetWindowUserPointer(window));

    if (!self) return;
    if (self->key_callbacks.empty()) return;

    for (const auto& [key_, callback] : self->key_callbacks)
    {
        callback(key, scancode, action, mods);
    }
}
