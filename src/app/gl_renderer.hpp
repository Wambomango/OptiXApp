#pragma once

#include "scene.hpp"
#include "window.hpp"
#include "camera.hpp"
#include "ssao.hpp"

#include "utils/opengl/shader.hpp"
#include "utils/opengl/vertex_array.hpp"
#include "utils/opengl/buffer.hpp"

#include <glad/glad.h>

class GLRenderer 
{
    public:
        GLRenderer(Window &window, Scene& scene);
        ~GLRenderer();

        void Render(Camera &camera);

    private:

        int width;
        int height;
        GLuint output_texture;

        GLuint g_position_texture;
        GLuint g_normal_texture;
        GLuint g_depth_texture;
        GLuint g_framebuffer;

        GLuint deferred_framebuffer;
        std::unique_ptr<OpenGL::Buffer> scene_buffer;
        OpenGL::VertexArray prepass_vao;
        OpenGL::VertexArray deferred_vao;
        std::unique_ptr<OpenGL::Program> prepass_program;
        std::unique_ptr<OpenGL::Program> deferred_program;
        int n_vertices = 0;

        SSAO ssao;
};