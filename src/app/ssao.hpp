#pragma once

#include "utils/opengl/buffer.hpp"
#include "utils/opengl/shader.hpp"
#include "utils/opengl/texture.hpp"

#include "camera.hpp"
#include "bindings.hpp"

class SSAO
{
    public:
        SSAO(int width, int height);
        ~SSAO();
        void CalculateSSAO(Camera &camera);

        std::unique_ptr<OpenGL::Buffer> samples;

    private:
        int width;
        int height;

        GLuint ssao_raw_texture;
        GLuint ssao_texture;
        GLuint ssao_raw_framebuffer;
        GLuint ssao_framebuffer;

        std::unique_ptr<OpenGL::Program> ssao_program;
        std::unique_ptr<OpenGL::Program> ssao_blur_program;

};