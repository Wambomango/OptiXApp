#pragma once

#include "utils/opengl/context.hpp"
#include "utils/opengl/shader.hpp"
#include "utils/opengl/buffer.hpp"
#include "utils/opengl/vertex_array.hpp"
#include "utils/opengl/texture.hpp"


#include <string>

#include "bindings.hpp"

class Window
{
    public:

        Window(int width, int height, std::string titel);
        ~Window();
        float GetTime();
        void SetScrollCallback(std::function<void(double, double)> callback);
        void SetMouseMoveCallback(std::function<void(double, double)> callback);
        void SetMouseButtonCallback(std::function<void(int, int, int)> callback);
        void SetKeyCallback(std::function<void(int, int, int, int)> callback);
        GLuint GetTexture();
        int GetWidth();
        int GetHeight();
        void Resize(int new_width, int new_height);
        bool ShouldClose();
        void Render();
        void SwapBuffers();
        void PollEvents();


    private:
        OpenGL::ContextGLFW context;
        int width;
        int height;
        std::string titel;

        std::unique_ptr<OpenGL::Program> screenquad_program;
        std::unique_ptr<OpenGL::VertexArray> screenquad_vao;

        GLuint output_texture;

        std::function<void(double, double)> scroll_callback = nullptr;
        std::function<void(double, double)> mouse_move_callback = nullptr;
        std::function<void(int, int, int)> mouse_button_callback = nullptr;
        std::function<void(int, int, int, int)> key_callback = nullptr;
};


