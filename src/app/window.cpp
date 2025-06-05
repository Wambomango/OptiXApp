#include "window.hpp"


Window::Window(int width, int height, std::string titel) : context(nullptr, true, true), width(width), height(height), titel(titel)
{
    context.Resize(width, height);
    context.Activate();

    screenquad_vao = std::make_unique<OpenGL::VertexArray>();
    screenquad_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("screenquad.vert"), SHADER_DIR + std::string("screenquad.frag"));

    GL_CREATE_TEXTURE_2D(output_texture, GL_RGBA8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_OUTPUT, output_texture);
}

Window::~Window()
{
    glDeleteTextures(1, &output_texture);
}

float Window::GetTime()
{
    return static_cast<float>(glfwGetTime());
}

void Window::SetScrollCallback(std::function<void(double, double)> callback)
{
    scroll_callback = callback;
    glfwSetWindowUserPointer(context.window, this);

    glfwSetScrollCallback(context.window, [](GLFWwindow* window, double xoffset, double yoffset)
    {
        auto self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self && self->scroll_callback)
        {
            self->scroll_callback(xoffset, yoffset);
        }
    });
}
        
void Window::SetMouseMoveCallback(std::function<void(double, double)> callback)
{
    mouse_move_callback = callback;
    glfwSetWindowUserPointer(context.window, this);

    glfwSetCursorPosCallback(context.window, [](GLFWwindow* window, double xpos, double ypos)
    {
        auto self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self && self->mouse_move_callback)
        {
            self->mouse_move_callback(xpos, ypos);
        }
    });
}

void Window::SetMouseButtonCallback(std::function<void(int, int, int)> callback)
{
    mouse_button_callback = callback;
    glfwSetWindowUserPointer(context.window, this);

    glfwSetMouseButtonCallback(context.window, [](GLFWwindow* window, int button, int action, int mods)
    {
        auto self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self && self->mouse_button_callback)
        {
            self->mouse_button_callback(button, action, mods);
        }
    });
}

void Window::SetKeyCallback(std::function<void(int, int, int, int)> callback)
{
    key_callback = callback;
    glfwSetWindowUserPointer(context.window, this);

    glfwSetKeyCallback(context.window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        auto self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self && self->key_callback)
        {
            self->key_callback(key, scancode, action, mods);
        }
    });
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
