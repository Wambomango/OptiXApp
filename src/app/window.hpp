#pragma once

#include "utils/opengl/context.hpp"
#include "utils/opengl/shader.hpp"
#include "utils/opengl/buffer.hpp"
#include "utils/opengl/vertex_array.hpp"
#include "utils/opengl/texture.hpp"

#include <map>
#include <string>

#include "bindings.hpp"

class Window
{
    public:

        Window(int width, int height, std::string titel);
        ~Window();
        float GetTime();
        int AddScrollCallback(std::function<void(double, double)> callback);
        int AddMouseMoveCallback(std::function<void(double, double)> callback);
        int AddMouseButtonCallback(std::function<void(int, int, int)> callback);
        int AddKeyCallback(std::function<void(int, int, int, int)> callback);
        bool RemoveScrollCallback(int key);
        bool RemoveMouseMoveCallback(int key);
        bool RemoveMouseButtonCallback(int key);
        bool RemoveKeyCallback(int key);
        GLuint GetTexture();
        int GetWidth();
        int GetHeight();
        void Resize(int new_width, int new_height);
        bool ShouldClose();
        void Render();
        void SwapBuffers();
        void PollEvents();


    private:
        static void ScrollDispatcher(GLFWwindow* window, double xoffset, double yoffset);
        static void MouseMoveDispatcher(GLFWwindow* window, double xpos, double ypos);
        static void MouseButtonDispatcher(GLFWwindow* window, int button, int action, int mods);
        static void KeyDispatcher(GLFWwindow* window, int key, int scancode, int action, int mods);

        OpenGL::ContextGLFW context;
        int width;
        int height;
        std::string titel;

        std::unique_ptr<OpenGL::Program> screenquad_program;
        std::unique_ptr<OpenGL::VertexArray> screenquad_vao;

        GLuint output_texture;

        std::map<int, std::function<void(double, double)>> scroll_callbacks;
        std::map<int, std::function<void(double, double)>> mouse_move_callbacks;
        std::map<int, std::function<void(int, int, int)>> mouse_button_callbacks;
        std::map<int, std::function<void(int, int, int, int)>> key_callbacks;
};


