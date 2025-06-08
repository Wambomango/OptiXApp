#include "ssao.hpp"

#include <random>

#define N_SAMPLES 999983

SSAO::SSAO(int width, int height) : width(width), height(height)
{
    GL_CREATE_TEXTURE_2D(ssao_raw_texture, GL_R8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_SSAO_RAW, ssao_raw_texture);

    GL_CREATE_TEXTURE_2D(ssao_texture, GL_R8, width, height);
    glBindTextureUnit(TEXTURE_UNIT_SSAO, ssao_texture);

    glCreateFramebuffers(1, &ssao_raw_framebuffer);
    glNamedFramebufferTexture(ssao_raw_framebuffer, GL_COLOR_ATTACHMENT0, ssao_raw_texture, 0);
    glCheckNamedFramebufferStatus(ssao_raw_framebuffer, GL_FRAMEBUFFER);

    glCreateFramebuffers(1, &ssao_framebuffer);
    glNamedFramebufferTexture(ssao_framebuffer, GL_COLOR_ATTACHMENT0, ssao_texture, 0);
    glCheckNamedFramebufferStatus(ssao_framebuffer, GL_FRAMEBUFFER);
    
    ssao_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("ssao.vert"), SHADER_DIR + std::string("ssao.frag"));
    ssao_program->SetInt("n_samples", N_SAMPLES);
    ssao_program->SetInt("width", width);
    ssao_program->SetInt("height", height);

    ssao_blur_program = std::make_unique<OpenGL::Program>(SHADER_DIR + std::string("ssao_blur.vert"), SHADER_DIR + std::string("ssao_blur.frag"));
    ssao_blur_program->SetInt("width", width);
    ssao_blur_program->SetInt("height", height);

    samples = std::make_unique<OpenGL::Buffer>(N_SAMPLES * 3 * sizeof(float), GL_STATIC_DRAW);
    std::vector<float> samples_cpu(N_SAMPLES * 3);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for(int i = 0; i < N_SAMPLES; i++) 
    {
        glm::vec3 sample;
        while(true)
        {
            sample.x = distribution(generator);
            sample.y = distribution(generator);
            sample.z = distribution(generator);

            if(glm::length(sample) < 1.0f) 
            {
                break;
            }
        }

        samples_cpu[i * 3 + 0] = sample.x;
        samples_cpu[i * 3 + 1] = sample.y;
        samples_cpu[i * 3 + 2] = sample.z;
    }

    samples->Store(samples_cpu.data(), N_SAMPLES * 3 * sizeof(float));    
    samples->BindRange(SSBO_SAMPLES);
}


SSAO::~SSAO()
{
    ssao_program.release();
    ssao_blur_program.release();

    glDeleteTextures(1, &ssao_raw_texture);
    glDeleteTextures(1, &ssao_texture);
    glDeleteFramebuffers(1, &ssao_raw_framebuffer);
    glDeleteFramebuffers(1, &ssao_framebuffer);
}


void SSAO::CalculateSSAO(Camera &camera)
{
    glBindFramebuffer(GL_FRAMEBUFFER, ssao_raw_framebuffer);
    ssao_program->SetMat4("view", 1, camera.GetViewMatrix());
    ssao_program->SetMat4("projection", 1, camera.GetProjectionMatrix());
    ssao_program->SetFloat("near_plane", camera.GetNearPlane());
    ssao_program->SetFloat("far_plane", camera.GetFarPlane());
    ssao_program->DrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, ssao_framebuffer);
    ssao_blur_program->DrawArrays(GL_TRIANGLES, 0, 3);
}

