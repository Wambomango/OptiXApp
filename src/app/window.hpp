#pragma once

#include "utils/opengl/context.hpp"
#include "utils/opengl/shader.hpp"
#include "utils/opengl/buffer.hpp"
#include "utils/opengl/vertex_array.hpp"

#include <string>


class Window
{
    public:

        Window(int width, int height, std::string titel) : context(nullptr, true, true), width(width), height(height), titel(titel)
        {
            context.Resize(width, height);
            context.Activate();

            screenquad_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("screenquad.vert"), SHADER_DIR + std::string("screenquad.frag"));
            screenquad_vao = std::make_unique<OpenGL::VertexArray>();

            glCreateTextures(GL_TEXTURE_2D, 1, &gl_texture);
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTextureUnit(1, gl_texture); 
            glTextureStorage2D(gl_texture, 1, GL_RGBA8, width, height);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        ~Window()
        {
            glDeleteTextures(1, &gl_texture);
        }

        void SetScrollCallback(std::function<void(double, double)> callback)
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
        
        void SetMouseMoveCallback(std::function<void(double, double)> callback)
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

        void SetMouseButtonCallback(std::function<void(int, int, int)> callback)
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

        void SetKeyCallback(std::function<void(int, int, int, int)> callback)
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

        GLuint GetTexture()
        {
            return gl_texture;
        }

        int GetWidth()
        {
            return width;
        }

        int GetHeight()
        {
            return height;
        }
        
        void Resize(int new_width, int new_height)
        {
            width = new_width;
            height = new_height;
            context.Resize(width, height);
        }

        bool ShouldClose()
        {
            return glfwWindowShouldClose(context.window);
        }

        void Render()
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(0, 0, width, height);

            screenquad_program->Use();
            screenquad_vao->Bind();
            glDrawArrays(GL_TRIANGLES, 0, 3);
            screenquad_vao->Unbind();
        }

        void SwapBuffers()
        {
            glfwSwapBuffers(context.window);
        }

        void PollEvents()
        {
            glfwPollEvents();
        }


    private:
        OpenGL::ContextGLFW context;
        int width;
        int height;
        std::string titel;

        std::unique_ptr<OpenGL::Program> screenquad_program;
        std::unique_ptr<OpenGL::VertexArray> screenquad_vao;

        GLuint gl_texture;

        std::function<void(double, double)> scroll_callback = nullptr;
        std::function<void(double, double)> mouse_move_callback = nullptr;
        std::function<void(int, int, int)> mouse_button_callback = nullptr;
        std::function<void(int, int, int, int)> key_callback = nullptr;
};


