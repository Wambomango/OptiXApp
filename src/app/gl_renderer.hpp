#pragma once

#include "scene.hpp"
#include "window.hpp"
#include "camera.hpp"

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

        GLuint scene_depth_texture;
        GLuint scene_framebuffer;
        std::unique_ptr<OpenGL::Buffer> scene_buffer;
        OpenGL::VertexArray scene_vao;
        std::unique_ptr<OpenGL::Program> scene_program;
        int n_vertices = 0;
};